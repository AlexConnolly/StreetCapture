"""v2 intelligence layer: groups, entities, label propagation, notifications.

Everything here consumes the artifact embeddings we already collect:

* recluster()          - unsupervised clustering -> "suggested" groups to name
* create_from_text()   - zero-shot: a text concept -> a labeled group
* create_from_artifact - one example -> a labeled group (its neighbours)
* search()             - type a concept, rank artifacts (CLIP text->image)
* similar()            - nearest neighbours of an artifact
* on_new_artifact()    - the LEARNING loop: every new artifact is matched to
                         labeled groups (auto-tag + optional notify) and to
                         entities (same instance over time)

Centroids are image-space means of L2-normalised embeddings, so cosine == dot.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ..taxonomy import category
from . import notify


def _norm(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n else v


# Concepts we try to match a cluster's centroid against (CLIP zero-shot) so an
# "unnamed cluster" comes with a human hint of what it probably is.
HINT_VOCAB = [
    "a pedestrian walking", "a person standing", "a person with a dog",
    "a cyclist on a bicycle", "a delivery driver", "a person in hi-vis",
    "a runner jogging", "a child", "a group of people",
    "a car", "a white van", "a delivery van", "a truck", "a bus",
    "a motorcycle", "a bicycle", "a taxi", "an emergency vehicle",
    "a dog", "a cat", "a bird", "a parked car", "a moving car",
    "a package or parcel", "a pushchair or pram", "an umbrella",
]


class GroupService:
    def __init__(self, cfg, db, embedder, vectorstore, reid=None):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.reid = reid     # person-ReID embedder (identity); None = CLIP fallback
        self.vs = vectorstore
        self._lock = threading.Lock()
        self._labeled = []   # [(gid, name, notify, last_notified, np_centroid, match_threshold)]
        self._seeded = set() # gids taught by drawn regions (use precise threshold)
        self._group_allowed_classes = {} # gid -> set(allowed classes)
        self._ents = []      # [[eid, occ, np_centroid, space], ...]
        self._web_base = ""  # set by the server for notification click-through
        self._vocab = None   # (labels, np matrix) cache for cluster hints

    # -- startup -----------------------------------------------------------
    def startup(self, web_base: str = "") -> None:
        self._web_base = web_base
        self._load_entities()
        self._refresh_labeled()
        # Person entities need a one-time ReID rebuild the first time this runs
        # (CLIP-blob entities -> clean per-person). It's ~1 min of CPU work over
        # all person crops, so do it off the startup path in a daemon thread.
        needs_reid = bool(self.reid and self.reid.enabled
                          and self.db.artifacts_missing_reid())
        if needs_reid:
            threading.Thread(target=self._safe_rebuild, name="ReIDRebuild",
                             daemon=True).start()
        else:
            try:
                n = self.backfill_entities()
                if n:
                    print(f"[groups] entity backfill: processed {n} historical artifacts")
            except Exception as e:
                print(f"[groups] entity backfill skipped: {e}")
        # If there are no groups at all yet, seed suggestions from what we have.
        if not self.db.list_groups():
            try:
                self.recluster()
            except Exception as e:
                print(f"[groups] initial recluster skipped: {e}")

        # Delete old manual user-defined groups that are not key-value tags
        try:
            with self.db._lock:
                self.db._conn.execute(
                    "DELETE FROM groups WHERE kind='labeled' AND (tag_key IS NULL OR tag_key='' OR tag_value IS NULL OR tag_value='')"
                )
                self.db._conn.execute(
                    "DELETE FROM group_members WHERE group_id NOT IN (SELECT id FROM groups)"
                )
                self.db._conn.commit()
        except Exception as e:
            print(f"[groups] failed to clean up old manual groups: {e}")

        # Upgrade existing labeled groups to the discriminative centroid model
        try:
            labeled = [g for g in self.db.list_groups() if g["kind"] == "labeled" and g.get("tag_key") and g.get("tag_value")]
            swept = 0
            for g in labeled:
                self._recompute_centroid(g["id"])
                # Sweep out anything the machine auto-tagged that no longer clears
                # the (now margin-based) threshold — cleans up low-confidence
                # mistakes (e.g. women previously tagged male) without manual work.
                swept += self.reclassify_group(g["id"], retrain=False).get("removed", 0)
                self._backfill_group(g["id"])
            if labeled:
                print(f"[groups] startup: upgraded {len(labeled)} labeled groups "
                      f"(reject-sweep untagged {swept} low-confidence auto-tags)")
        except Exception as e:
            print(f"[groups] startup group upgrade failed: {e}")

        # Start background recluster daemon
        self._stop_event = threading.Event()
        self._recluster_thread = threading.Thread(
            target=self._recluster_loop, name="BackgroundRecluster", daemon=True
        )
        self._recluster_thread.start()

    def _safe_rebuild(self) -> None:
        try:
            self.rebuild_entities()
        except Exception as e:  # noqa: BLE001
            print(f"[groups] entity rebuild failed: {e}")

    def _recluster_loop(self) -> None:
        last_embedding_count = 0
        try:
            with self.db._lock:
                last_embedding_count = self.db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        except Exception:
            pass

        # Wait a bit on startup
        time.sleep(10)
        while not self._stop_event.is_set():
            try:
                with self.db._lock:
                    count = self.db._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                if count != last_embedding_count:
                    print(f"[groups] background recluster: embedding count changed ({last_embedding_count} -> {count}), reclustering...")
                    self.recluster()
                    last_embedding_count = count
            except Exception as e:
                print(f"[groups] background recluster error: {e}")
            
            # Sleep 60 seconds, checking for stop event in 1s steps
            for _ in range(60):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def stop(self) -> None:
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_recluster_thread"):
            self._recluster_thread.join(timeout=2)

    def _entity_inputs(self, artifact_id, cls, clip_vec):
        """Return (vector, space, threshold) for entity resolution. ReID (which
        actually separates object instances) for everything it can embed; CLIP
        only as a fallback — CLIP over-merges different cars/people into blobs."""
        if self.reid and self.reid.enabled:
            r = self._artifact_reid(artifact_id)
            if r is not None:
                return _norm(r), "reid", self.cfg.reid_entity_threshold
        return _norm(clip_vec), "clip", self.cfg.entity_threshold

    def reject_entity_member(self, artifact_id: int) -> dict:
        """User says this artifact is NOT the same as the rest of its entity:
        record 'don't merge' constraints against the other members and split it
        into its own entity. Constraints survive rebuilds."""
        eid = self.db.entity_of(artifact_id)
        if eid is None:
            return {"error": "not in an entity"}
        others = [m for m in self.db.entity_members(eid) if m != artifact_id]
        for m in others:
            self.db.add_dislink(artifact_id, m)
        rv = self._artifact_reid(artifact_id)
        space = "reid"
        if rv is None:
            rv = self.db.embedding_for(artifact_id)
            space = "clip"
        if rv is None:
            return {"error": "no embedding"}
        with self._lock:
            new_eid = self.db.insert_entity(_norm(rv).tolist(), time.time(), space)
            self.db.set_artifact_entity(artifact_id, new_eid)
        self._load_entities()
        return {"ok": True, "new_entity": new_eid, "dislinks": len(others)}

    def _artifact_reid(self, artifact_id):
        """ReID vector for an artifact's crop (cached in the DB)."""
        r = self.db.reid_for(artifact_id)
        if r is not None:
            return r
        path = self.db.image_path(artifact_id, 0)
        if not path:
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        r = self.reid.embed(img)
        if r is not None:
            self.db.insert_reid(artifact_id, r)
        return r

    def backfill_entities(self) -> int:
        rows = self.db.embeddings_missing_entity_with_class()
        for aid, vec, cls in rows:
            v, space, thr = self._entity_inputs(aid, cls, vec)
            self._assign_entity(aid, v, space, thr)
        return len(rows)

    def rebuild_entities(self) -> dict:
        """Rebuild all entities from ReID with COMPLETE-LINKAGE agglomerative
        clustering. Unlike greedy running-mean centroids (which drift to a generic
        average and snowball visibly-different people together), complete linkage
        only merges items that are close to EVERY member — so a striped-shirt
        person can't be pulled into a cluster she doesn't actually resemble."""
        from sklearn.cluster import AgglomerativeClustering

        t0 = time.time()
        missing = self.db.artifacts_missing_reid()
        for aid, path in missing:
            img = cv2.imread(str(path))
            if img is not None:
                r = self.reid.embed(img)
                if r is not None:
                    self.db.insert_reid(aid, r)
        if missing:
            print(f"[groups] ReID: embedded {len(missing)} crops in {time.time()-t0:.0f}s")

        data = self.db.all_reid_with_times()
        if not data:
            return {"artifacts": 0, "entities": 0}
        ids = [d[0] for d in data]
        X = np.stack([_norm(d[1]) for d in data])
        if len(X) < 2:
            labels = np.zeros(len(X), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.cfg.entity_cluster_distance,
                metric="cosine", linkage="complete").fit_predict(X)
        labels = self._apply_dislinks(labels, ids)

        # group -> entity
        by = {}
        for i, lab in enumerate(labels):
            by.setdefault(int(lab), []).append(i)
        entities, assigns = [], []
        for k, (lab, idxs) in enumerate(by.items()):
            centroid = _norm(X[idxs].mean(axis=0))
            fs = min(data[i][2] for i in idxs)
            ls = max(data[i][3] or data[i][2] for i in idxs)
            entities.append({"centroid": centroid.tolist(), "space": "reid",
                             "occ": len(idxs), "first_seen": fs, "last_seen": ls})
            for i in idxs:
                assigns.append((ids[i], k))

        self.db.reset_entities()
        self.db.rebuild_entities_write(entities, assigns, time.time())
        self._load_entities()
        print(f"[groups] entity rebuild: {len(data)} artifacts -> {len(entities)} "
              f"entities (complete-linkage) in {time.time()-t0:.0f}s")
        return {"artifacts": len(data), "entities": len(entities)}

    def _apply_dislinks(self, labels, ids):
        """Split any 'not the same' pair that landed in the same cluster."""
        dislinks = self.db.all_dislinks()
        if not dislinks:
            return labels
        labels = np.array(labels, dtype=int)
        pos = {aid: i for i, aid in enumerate(ids)}
        nxt = int(labels.max()) + 1 if len(labels) else 0
        for _ in range(5):                       # iterate to convergence
            moved = False
            for a, b in dislinks:
                ia, ib = pos.get(a), pos.get(b)
                if ia is not None and ib is not None and labels[ia] == labels[ib]:
                    labels[ib] = nxt          # pull b out into its own cluster
                    nxt += 1
                    moved = True
            if not moved:
                break
        return labels

    def _refresh_labeled(self) -> None:
        rows = self.db.labeled_group_centroids()
        self._labeled = [(g, n, notify, ln, _norm(c), mthr)
                         for (g, n, notify, ln, c, mthr) in rows]
        self._seeded = self.db.seeded_group_ids()
        self._group_allowed_classes = {
            gid: self.db.group_allowed_classes(gid)
            for (gid, _, _, _, _, _) in self._labeled
        }

    def _load_entities(self) -> None:
        self._ents = [[eid, occ, _norm(c), space]
                      for (eid, occ, c, space) in self.db.entity_centroids()]

    # -- embeddings matrix -------------------------------------------------
    def _matrix(self, rows=None):
        if rows is None:
            rows = self.db.all_embeddings()
        if not rows:
            return [], np.zeros((0, 1), "float32"), []
        ids = [r[0] for r in rows]
        classes = [r[2] for r in rows]
        X = np.stack([_norm(r[1]) for r in rows])
        return ids, X, classes

    # -- clustering --------------------------------------------------------
    def recluster(self) -> dict:
        # Cluster only artifacts not already confirmed into a labeled group, so
        # suggestions shrink as the user categorises instead of endlessly
        # re-proposing things they've already handled.
        ids, X, classes = self._matrix(self.db.embeddings_for_clustering())
        if len(ids) < self.cfg.cluster_min_size:
            return {"clusters": 0, "artifacts": len(ids)}

        # Partition indices by primary class to prevent class mixing
        by_class = {}
        for i, cls in enumerate(classes):
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(i)

        clusters_to_save = []
        for cls, indices in by_class.items():
            if len(indices) < self.cfg.cluster_min_size:
                continue
            X_cls = X[indices]
            
            # Subspace Orthogonalization:
            # 1. Compute category mean vector representing the generic class shape
            mean_vec = _norm(X_cls.mean(axis=0))
            
            # 2. Subtract component of each embedding that points along the mean vector
            projections = (X_cls @ mean_vec)[:, np.newaxis] * mean_vec
            residuals = X_cls - projections
            
            # 3. Calculate residual norms and filter out generic baseline objects (norm < 0.30)
            # This avoids amplifying random noise on standard objects that are close to the category mean.
            residual_norms = np.linalg.norm(residuals, axis=1)
            distinctive_idx = np.where(residual_norms >= 0.30)[0]
            if len(distinctive_idx) < self.cfg.cluster_min_size:
                continue

            # Extract the subset of distinctive residuals and normalize them
            residuals_distinct = residuals[distinctive_idx]
            norms_distinct = residual_norms[distinctive_idx][:, np.newaxis]
            R = residuals_distinct / norms_distinct

            # 4. Compute Pairwise Similarity Matrix
            S = R @ R.T  # shape: (M, M) where M = len(distinctive_idx)
            M = len(distinctive_idx)
            
            # K-nearest neighbors (including self)
            K = self.cfg.cluster_min_size  # e.g., 5
            if M < K:
                continue
                
            candidates = []
            for i in range(M):
                # Sort neighbors by similarity DESC
                neighbors = np.argsort(-S[i])[:K]  # top K neighbors indices in distinctive space
                
                # Evaluate Cohesion: average pairwise similarity of neighbors
                sub_S = S[neighbors][:, neighbors]
                cohesion = float(sub_S.mean())
                
                # Evaluate Deviation: distance of neighborhood's original centroid to mean_vec
                global_neighbors = [indices[distinctive_idx[n]] for n in neighbors]
                centroid_orig = _norm(X[global_neighbors].mean(axis=0))
                deviation = float(1.0 - (centroid_orig @ mean_vec))
                
                # Check threshold criteria:
                # - cohesion >= 0.45 (highly cohesive)
                # - deviation >= 0.20 (distinct from average category)
                if cohesion >= 0.45 and deviation >= 0.20:
                    candidates.append({
                        "indices": set(global_neighbors),
                        "deviation": deviation,
                        "centroid": centroid_orig
                    })
                    
            if not candidates:
                continue
                
            # Sort candidates by deviation DESC (most distinct first)
            candidates.sort(key=lambda c: -c["deviation"])
            
            # 5. Greedily merge overlapping neighborhoods
            merged_clusters = []
            for cand in candidates:
                merged = False
                for existing in merged_clusters:
                    # If candidate overlaps significantly with existing (>= 40% overlap of candidate size)
                    intersection = len(cand["indices"] & existing["indices"])
                    if intersection / len(cand["indices"]) >= 0.40:
                        existing["indices"].update(cand["indices"])
                        merged = True
                        break
                if not merged:
                    merged_clusters.append({
                        "indices": cand["indices"],
                        "deviation": cand["deviation"],
                        "centroid": cand["centroid"]
                    })
                    
            # 6. Save the final distinct clusters
            for cluster in merged_clusters:
                c_indices = list(cluster["indices"])
                if len(c_indices) < self.cfg.cluster_min_size:
                    continue
                    
                # Recompute final centroid for the merged cluster
                centroid = _norm(X[c_indices].mean(axis=0))
                deviation = float(1.0 - (centroid @ mean_vec))
                
                # Prepare members list (dot product to centroid in original space)
                members = [
                    (ids[i], float(X[i] @ centroid))
                    for i in c_indices
                ]
                
                # Sort members by similarity to centroid DESC
                members.sort(key=lambda m: -m[1])
                
                # Get hint
                hint = self._hint_for(centroid, [classes[i] for i in c_indices])
                
                clusters_to_save.append({
                    "centroid": centroid.tolist(),
                    "size": len(c_indices),
                    "hint": hint,
                    "deviation": deviation,
                    "members": members
                })

        self.db.recluster_save(clusters_to_save)
        return {"clusters": len(clusters_to_save), "artifacts": len(ids)}

    def _vocab_matrix(self):
        """Lazily embed the hint vocabulary once (CLIP text space)."""
        if self._vocab is not None:
            return self._vocab
        vecs = []
        for phrase in HINT_VOCAB:
            tv = self.embedder.embed_text(phrase)
            vecs.append(_norm(tv) if tv is not None else None)
        if any(v is None for v in vecs):
            self._vocab = ([], None)
        else:
            self._vocab = (HINT_VOCAB, np.stack(vecs))
        return self._vocab

    def _hint_for(self, centroid, member_classes) -> str | None:
        """Best concept for a cluster centroid, falling back to dominant class."""
        if member_classes:
            top = max(set(member_classes), key=member_classes.count)
            if top == "person":
                return "people"
            return f"{top}s"
        return "objects"

    # -- labeled group creation -------------------------------------------
    def name_cluster(self, group_id: int, name: str) -> None:
        """Promote a cluster suggestion to a labeled (learning) group."""
        vecs = self.db.group_member_vectors(group_id)
        centroid = _norm(np.stack([_norm(v) for v in vecs]).mean(axis=0)) if vecs else None

        # Parse "key: value" format into tag_key/tag_value for structured tagging
        tag_key, tag_value = None, None
        if ":" in name:
            parts = name.split(":", 1)
            tag_key = parts[0].strip()
            tag_value = parts[1].strip()

        self.db.update_group(group_id, name=name, kind="labeled",
                             centroid=centroid.tolist() if centroid is not None else None,
                             size=len(vecs), tag_key=tag_key, tag_value=tag_value)
        # Auto-confirm all current members so they serve as the basis for learning/auto-tagging
        member_ids = self.db.group_members(group_id)
        if member_ids:
            self.set_members_feedback(group_id, member_ids, "confirmed")
            # Recompute the discriminative centroid from the confirmed members and
            # backfill, so the newly-named group learns and starts auto-tagging.
            self.train_and_backfill(group_id)
        else:
            self._refresh_labeled()

    def tag_artifacts(self, artifact_ids: list[int], tags: list[dict],
                      source_group_id: int | None = None) -> dict:
        """Tag a list of artifacts with multiple key-value tags, creating/promoting tag groups."""
        if not tags:
            return {"error": "no tags provided"}

        group_ids = []
        for t in tags:
            tag_key = t.get("key", "").strip()
            tag_value = t.get("value", "").strip()
            if not tag_key or not tag_value:
                continue

            # 1. Find or create the labeled tag group
            with self.db._lock:
                row = self.db._conn.execute(
                    "SELECT id FROM groups WHERE kind='labeled' AND tag_key=? AND tag_value=? COLLATE NOCASE",
                    (tag_key, tag_value)
                ).fetchone()
                
            if row:
                group_id = row[0]
            else:
                name = f"{tag_key}: {tag_value}"
                group_id = self.db.insert_group(
                    name=name, kind="labeled", centroid=None, size=0,
                    tag_key=tag_key, tag_value=tag_value
                )
            group_ids.append(group_id)

            # 2. Add selected artifacts as confirmed members of the tag group
            self.set_members_feedback(group_id, artifact_ids, "confirmed")

        # 3. Compute each tag group's centroid from its confirmed members and
        #    backfill existing artifacts. Without this a freshly-created tag
        #    group keeps a NULL centroid, so labeled_group_centroids() (and thus
        #    _labeled / _auto_tag) skips it and NEW artifacts are never tagged.
        for group_id in group_ids:
            self.train_and_backfill(group_id)

        # 4. Clean up source group (suggested cluster) if provided
        if source_group_id is not None and group_ids:
            with self.db._lock:
                self.db._conn.execute(
                    "DELETE FROM group_members WHERE group_id=? AND artifact_id IN (" +
                    ",".join("?" for _ in artifact_ids) + ")",
                    [source_group_id] + list(artifact_ids)
                )
                rem_count = self.db._conn.execute(
                    "SELECT COUNT(*) FROM group_members WHERE group_id=? AND (status IS NULL OR status!='rejected')",
                    (source_group_id,)
                ).fetchone()[0]
                
                if rem_count == 0:
                    self.db._conn.execute("DELETE FROM groups WHERE id=?", (source_group_id,))
                    self.db._conn.execute("DELETE FROM group_members WHERE group_id=?", (source_group_id,))
                else:
                    self.db._conn.execute(
                        "UPDATE groups SET size=? WHERE id=?", (rem_count, source_group_id)
                    )
                self.db._conn.commit()

        self._refresh_labeled()
        return {"status": "ok", "group_ids": group_ids}

    def get_tags_autocomplete(self) -> dict:
        """Return unique keys and their matching values for autocomplete.
        Sources from both labeled groups (tag_key/tag_value) and the labels table (type/value)."""
        with self.db._lock:
            # 1. From labeled groups
            group_rows = self.db._conn.execute(
                "SELECT DISTINCT tag_key, tag_value FROM groups "
                "WHERE kind='labeled' AND tag_key IS NOT NULL AND tag_value IS NOT NULL"
            ).fetchall()
            # 2. From artifact labels (taxonomy)
            label_rows = self.db._conn.execute(
                "SELECT DISTINCT type, value FROM labels"
            ).fetchall()
            
        values: dict[str, list[str]] = {}
        for k, v in list(group_rows) + list(label_rows):
            if not k or not v:
                continue
            if k not in values:
                values[k] = []
            if v not in values[k]:
                values[k].append(v)
                
        # Sort everything
        for k in values:
            values[k] = sorted(values[k])
        keys = sorted(values.keys())
            
        return {"keys": keys, "values": values}

    def create_from_text(self, name: str, prompt: str, top: int = 25) -> dict:
        tvec = self.embedder.embed_text(prompt)
        if tvec is None:
            return {"error": "text encoder unavailable"}
        ids, X, _ = self._matrix()
        if not ids:
            return {"error": "no artifacts yet"}
        sims = X @ _norm(tvec)
        order = np.argsort(-sims)[:top]
        picked = [(ids[i], float(sims[i])) for i in order
                  if sims[i] >= self.cfg.text_match_threshold] or \
                 [(ids[i], float(sims[i])) for i in order[:5]]
        member_vecs = [X[i] for i in order[:len(picked)]]
        centroid = _norm(np.stack(member_vecs).mean(axis=0))
        gid = self.db.insert_group(name, "labeled", centroid.tolist(), len(picked))
        for aid, sc in picked:
            self.db.add_member(gid, aid, sc, "text")
        self._refresh_labeled()
        return {"group_id": gid, "members": len(picked)}

    def create_from_artifact(self, name: str, artifact_id: int, top: int = 25) -> dict:
        vec = self.db.embedding_for(artifact_id)
        if vec is None:
            return {"error": "artifact has no embedding"}
        ids, X, _ = self._matrix()
        sims = X @ _norm(vec)
        order = np.argsort(-sims)[:top]
        picked = [(ids[i], float(sims[i])) for i in order
                  if sims[i] >= self.cfg.group_match_threshold] or [(artifact_id, 1.0)]
        member_vecs = [X[i] for i in order[:len(picked)]]
        centroid = _norm(np.stack(member_vecs).mean(axis=0)) if member_vecs else _norm(vec)
        gid = self.db.insert_group(name, "labeled", centroid.tolist(), len(picked))
        for aid, sc in picked:
            source_type = "manual" if aid == artifact_id else "auto"
            self.db.add_member(gid, aid, sc, source_type)
            if aid == artifact_id:
                self.db.set_member_status(gid, aid, "confirmed")
        self._refresh_labeled()
        return {"group_id": gid, "members": len(picked)}

    def label_region(self, artifact_id: int, rank: int, box, label: str) -> dict:
        """Teach a custom label from a region the user drew on an image.

        Crops [x1,y1,x2,y2] (normalised 0-1) from the artifact's stored image,
        CLIP-embeds the crop, and adds it as a SEED of a labeled group named
        `label` (created if new). The group's prototype = mean of its seeds, so
        drawing more examples of the same label sharpens it and new artifacts
        get auto-tagged by similarity. Few-shot, no model training."""
        label = (label or "").strip()
        if not label:
            return {"error": "empty label"}

        vec = None
        if list(box) == [0.0, 0.0, 1.0, 1.0]:
            vec = self.db.embedding_for(artifact_id)

        if vec is None:
            path = self.db.image_path(artifact_id, rank)
            if not path:
                return {"error": "no image for that artifact"}
            img = cv2.imread(str(path))
            if img is None:
                return {"error": "image unreadable"}
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box
            px1, px2 = sorted((int(x1 * w), int(x2 * w)))
            py1, py2 = sorted((int(y1 * h), int(y2 * h)))
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)
            if px2 - px1 < 4 or py2 - py1 < 4:
                return {"error": "selection too small"}
            crop = img[py1:py2, px1:px2]
            vec = self.embedder.embed(crop)
            if vec is None:
                return {"error": "could not embed selection"}

        with self._lock:
            gid = self.db.group_id_by_name(label)
            created = gid is None
            if created:
                gid = self.db.insert_group(label, "labeled", vec, size=1)
            self.db.add_label_seed(gid, artifact_id, list(box), vec)
            # add the source artifact as a confirmed member too
            self.db.add_member(gid, artifact_id, 1.0, "seed")
            self.db.set_member_status(gid, artifact_id, "confirmed")
        self._recompute_centroid(gid)   # centroid = mean of seeds
        matched = self._backfill_group(gid)   # retro-apply to existing artifacts
        seeds = self.db.seed_count(gid)
        return {"group_id": gid, "label": label, "created": created,
                "seeds": seeds, "matched": matched}

    def _backfill_group(self, group_id: int, threshold: float | None = None) -> int:
        """Tag existing artifacts that already match this group's prototype, so a
        freshly-taught label is useful immediately (not only for future frames)."""
        entry = next(((c, mthr) for (g, _n, _no, _ln, c, mthr) in self._labeled
                      if g == group_id), None)
        if entry is None:
            return 0
        cent, mthr = entry
        ids, X, classes = self._matrix()
        if not ids:
            return 0
        allowed = self._group_allowed_classes.get(group_id, set())
        rejected = self.db.rejected_member_ids(group_id)
        existing = set(self.db.group_members(group_id))
        sims = X @ _norm(cent)

        if threshold is not None:
            thr = threshold
        elif mthr is not None:
            thr = mthr          # calibrated LR decision boundary
        else:
            thr = (self.cfg.label_match_threshold if group_id in self._seeded
                   else self.cfg.group_match_threshold)

        # Once a group is trusted (enough human confirmations), backfilled
        # matches are auto-classified rather than dumped into the verify queue.
        confident = self.db.human_confirmed_count(group_id) >= self.cfg.auto_classify_min_confirmed
        to_add = []
        for i in range(len(ids)):
            if allowed and classes[i] not in allowed:
                continue
            if sims[i] >= thr and ids[i] not in rejected and ids[i] not in existing:
                if confident:
                    to_add.append((ids[i], float(sims[i]), "auto_confirm", "confirmed"))
                else:
                    to_add.append((ids[i], float(sims[i]), "auto"))

        self.db.add_members(group_id, to_add)
        return len(to_add)

    def set_notify(self, group_id: int, on: bool) -> None:
        self.db.update_group(group_id, notify=1 if on else 0)
        self._refresh_labeled()

    def rename(self, group_id: int, name: str) -> None:
        tag_key, tag_value = None, None
        if ":" in name:
            parts = name.split(":", 1)
            tag_key = parts[0].strip()
            tag_value = parts[1].strip()
        self.db.update_group(group_id, name=name, tag_key=tag_key, tag_value=tag_value)
        self._refresh_labeled()

    def delete(self, group_id: int) -> None:
        self.db.delete_group(group_id)
        self._refresh_labeled()

    # -- live labelling ----------------------------------------------------
    def has_labels(self) -> bool:
        return bool(self._labeled)

    def match_live(self, crop, cls: str | None = None):
        """Best taught label for a live track crop, or None. Called (rate-limited)
        from the live loop — embeds the crop and matches labeled prototypes."""
        if not self._labeled:
            return None
        vec = self.embedder.embed(crop)
        if vec is None:
            return None
        v = _norm(vec)
        best_name, best_s = None, -1.0
        for (gid, name, _no, _ln, c, mthr) in self._labeled:
            if cls is not None:
                allowed = self._group_allowed_classes.get(gid, set())
                if allowed and cls not in allowed:
                    continue
            s = float(v @ c)
            if mthr is not None:
                thr = mthr          # calibrated LR decision boundary
            else:
                thr = (self.cfg.label_match_threshold if gid in self._seeded
                       else self.cfg.live_label_threshold)
            if s >= thr and s > best_s:
                best_name, best_s = name, s
        if best_name is not None:
            return {"label": best_name, "score": round(best_s, 3)}
        return None

    # -- search / similar --------------------------------------------------
    def search(self, prompt: str, k: int = 40) -> list[dict]:
        tvec = self.embedder.embed_text(prompt)
        if tvec is None:
            return []
        ids, X, classes = self._matrix()
        if not ids:
            return []
        sims = X @ _norm(tvec)
        order = np.argsort(-sims)[:k]
        return [{"artifact_id": ids[i], "score": round(float(sims[i]), 3),
                 "class": classes[i]} for i in order]

    def similar(self, artifact_id: int, k: int = 30) -> list[dict]:
        vec = self.db.embedding_for(artifact_id)
        if vec is None:
            return []
        ids, X, classes = self._matrix()
        sims = X @ _norm(vec)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if ids[i] == artifact_id:
                continue
            out.append({"artifact_id": ids[i], "score": round(float(sims[i]), 3),
                        "class": classes[i]})
            if len(out) >= k:
                break
        return out

    # -- the learning loop (called for every new artifact) ----------------
    def on_new_artifact(self, artifact_id: int, vec, cls: str) -> None:
        try:
            v = _norm(vec)
            ev, space, thr = self._entity_inputs(artifact_id, cls, vec)
            self._assign_entity(artifact_id, ev, space, thr)
            self._auto_tag(artifact_id, v, cls)   # taught labels stay CLIP
        except Exception as e:
            print(f"[groups] on_new_artifact error: {e}")

    def _assign_entity(self, artifact_id, v, space="clip", threshold=None) -> None:
        if threshold is None:
            threshold = self.cfg.entity_threshold
        now = time.time()
        with self._lock:
            best_i, best_s = -1, -1.0
            for i, (_eid, _occ, c, sp) in enumerate(self._ents):
                if sp != space:      # never compare across ReID / CLIP spaces
                    continue
                s = float(v @ c)
                if s > best_s:
                    best_i, best_s = i, s
            if best_i >= 0 and best_s >= threshold:
                eid, occ, c, sp = self._ents[best_i]
                occ2 = occ + 1
                c2 = _norm(c * occ + v)   # running-mean centroid, renormalised
                self._ents[best_i] = [eid, occ2, c2, sp]
                self.db.update_entity(eid, c2.tolist(), occ2, now)
                self.db.set_artifact_entity(artifact_id, eid)
            else:
                eid = self.db.insert_entity(v.tolist(), now, space)
                self._ents.append([eid, 1, v, space])
                self.db.set_artifact_entity(artifact_id, eid)

    def _auto_tag(self, artifact_id, v, cls) -> None:
        for i, (gid, name, notify_on, last_notified, c, mthr) in enumerate(list(self._labeled)):
            allowed = self._group_allowed_classes.get(gid, set())
            if allowed and cls not in allowed:
                continue
            s = float(v @ c)
            if mthr is not None:
                thr = mthr          # calibrated LR decision boundary
            else:
                thr = (self.cfg.label_match_threshold if gid in self._seeded
                       else self.cfg.group_match_threshold)
            if s < thr:
                continue
            # don't re-add / re-notify something the user explicitly rejected.
            if artifact_id in self.db.rejected_member_ids(gid):
                continue

            # Once a group has enough human confirmations it's trusted: matches
            # above its threshold are auto-classified (confirmed) with no verify
            # queue. Below that bar they stay as 'auto' suggestions to review.
            if self.db.human_confirmed_count(gid) >= self.cfg.auto_classify_min_confirmed:
                self.db.add_member(gid, artifact_id, s, "auto_confirm", "confirmed")
            else:
                self.db.add_member(gid, artifact_id, s, "auto")
            if notify_on and (time.time() - last_notified) >= self.cfg.notify_cooldown_s:
                self._notify(gid, name, artifact_id, cls, s)
                self.db.touch_group_notified(gid)
                self._labeled[i] = (gid, name, notify_on, time.time(), c, mthr)

    # -- feedback loop (curate a group -> improves its judgement) ----------
    def set_member_feedback(self, group_id: int, artifact_id: int, status: str) -> None:
        """status: 'confirmed' | 'rejected' | 'removed'."""
        if status == "removed":
            self.db.remove_member(group_id, artifact_id)
        elif status in ("confirmed", "rejected"):
            self.db.set_member_status(group_id, artifact_id, status)

    def set_members_feedback(self, group_id: int, artifact_ids: list[int], status: str) -> None:
        """Apply feedback status to multiple artifacts in batch.
        status: 'confirmed' | 'rejected' | 'removed'.
        """
        if artifact_ids:
            if status == "removed":
                for aid in artifact_ids:
                    self.db.remove_member(group_id, aid)
            elif status in ("confirmed", "rejected"):
                self.db.set_members_status(group_id, artifact_ids, status)

    def train_and_backfill(self, group_id: int) -> None:
        self._recompute_centroid(group_id)
        self._backfill_group(group_id)

    def _group_scorer(self, group_id: int):
        """(normalised centroid, match threshold) for a labeled group, or None."""
        entry = next(((c, mthr) for (g, _n, _no, _ln, c, mthr) in self._labeled
                      if g == group_id), None)
        if entry is None:
            return None
        cent, mthr = entry
        if mthr is not None:
            thr = mthr
        else:
            thr = (self.cfg.label_match_threshold if group_id in self._seeded
                   else self.cfg.group_match_threshold)
        return _norm(cent), thr

    def auto_classify_pending(self, group_id: int) -> dict:
        """'I've done enough training — let the model decide the rest.' Re-scores
        every un-reviewed suggestion against the current centroid: matches (>=
        threshold) are auto-classified (confirmed, source='auto_confirm'); the
        rest are dropped and DON'T get the tag. Nothing here trains the model
        (auto_confirm is excluded), so it just applies the model you already
        trained. Dropped ones aren't blocked — a later match can re-add them."""
        self._recompute_centroid(group_id)   # score against the latest model
        sc = self._group_scorer(group_id)
        if sc is None:
            return {"error": "group has no centroid"}
        cent, thr = sc

        classified, dropped = [], []
        for aid in self.db.pending_member_ids(group_id):
            vec = self.db.embedding_for(aid)
            s = float(_norm(vec) @ cent) if vec is not None else -1.0
            (classified if s >= thr else dropped).append(aid)

        # Matches carry the tag (source='auto_confirm' -> not used for training);
        # non-matches are dropped from the queue (a later match can re-add them).
        self.db.mark_auto_classified(group_id, classified)
        self.db.remove_members(group_id, dropped)
        return {"classified": len(classified), "dropped": len(dropped)}

    def reclassify_group(self, group_id: int, retrain: bool = True) -> dict:
        """A rejection is BAD news: relearn, then sweep. Re-scores every member
        the MACHINE auto-classified (source='auto_confirm') against the updated
        model and untags the ones that no longer clear the threshold — so
        rejecting one look-alike pulls its siblings out too. Your confirmations
        and seeds are never touched. Returns how many were swept out."""
        if retrain:
            self._recompute_centroid(group_id)
        sc = self._group_scorer(group_id)
        if sc is None:
            return {"removed": 0}
        cent, thr = sc
        to_remove = []
        for aid in self.db.auto_classified_member_ids(group_id):
            vec = self.db.embedding_for(aid)
            s = float(_norm(vec) @ cent) if vec is not None else -1.0
            if s < thr:
                to_remove.append(aid)
        self.db.remove_members(group_id, to_remove)
        if to_remove:
            print(f"[groups] reject-sweep: untagged {len(to_remove)} no-longer-matching "
                  f"auto-classified members from group {group_id}")
        return {"removed": len(to_remove)}

    def _recompute_centroid(self, group_id: int) -> dict:
        # Get positive vectors
        seeds = self.db.seed_vectors(group_id)
        if seeds:
            pos_vecs = seeds
            basis = "seeds"
        else:
            # Train on HUMAN-confirmed examples only — machine auto-classified
            # members (source='auto_confirm') must not drift the model toward its
            # own predictions.
            pos_vecs = self.db.group_member_vectors(
                group_id, status="confirmed", exclude_auto_confirm=True)
            basis = "confirmed"
            if not pos_vecs:
                pos_vecs = self.db.group_member_vectors(group_id, exclude_rejected=True)
                basis = "all"

        if not pos_vecs:
            return {"members": 0, "basis": basis}

        pos_mean = np.stack([_norm(x) for x in pos_vecs]).mean(axis=0)

        # Get explicit negative vectors (rejected members)
        neg_vecs = self.db.group_member_vectors(group_id, status="rejected")

        # Get primary classes of the positive members to sample background negatives
        member_ids = self.db.group_members(group_id)
        group_classes = set()
        allowed = self._group_allowed_classes.get(group_id, set())
        if allowed:
            group_classes = allowed
        else:
            if member_ids:
                with self.db._lock:
                    placeholders = ",".join("?" for _ in member_ids)
                    rows = self.db._conn.execute(
                        f"SELECT DISTINCT primary_class FROM artifacts WHERE id IN ({placeholders})",
                        tuple(member_ids)
                    ).fetchall()
                    group_classes = {r[0] for r in rows}

        # Fetch other artifacts of the same class to use as background negatives
        bg_vecs = self.db.background_negative_vectors(group_id, list(group_classes), limit=200) if group_classes else []

        # Combine all negatives
        all_neg_vecs = []
        if neg_vecs:
            all_neg_vecs.extend(neg_vecs)
        if bg_vecs:
            all_neg_vecs.extend(bg_vecs)

        # match_threshold: the cosine cut a matcher must clear against this
        # centroid. None => fall back to the global cosine defaults (correct for a
        # mean centroid). A discriminative LR centroid is a hyperplane NORMAL, not
        # a class mean — dot products against it live on a totally different (much
        # smaller) scale than the 0.72/0.78 cosine defaults, so it needs its own
        # calibrated cut or nothing ever matches it.
        match_threshold = None

        if all_neg_vecs and (len(pos_vecs) >= 5 or len(neg_vecs) > 0):
            # Train a discriminative Logistic Regression classifier
            # y=1 for positive group members, y=0 for background/rejected negatives
            X_pos = np.stack([_norm(x) for x in pos_vecs])
            X_neg = np.stack([_norm(x) for x in all_neg_vecs])

            X_train = np.vstack([X_pos, X_neg])
            y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))

            # Explicit rejections are LOUD: weight each one reject_weight× a
            # background negative so a single "no" actually rotates the boundary
            # away from the mistaken feature, instead of being 1-of-200.
            sample_weight = np.concatenate([
                np.ones(len(X_pos)),
                np.full(len(neg_vecs), self.cfg.reject_weight),   # rejected first
                np.ones(len(bg_vecs)),                            # then background
            ]) if all_neg_vecs else None

            from sklearn.linear_model import LogisticRegression
            # Train model with balanced weights to handle class imbalance
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=100, solver="liblinear")
            clf.fit(X_train, y_train, sample_weight=sample_weight)

            w = clf.coef_[0]
            b = float(clf.intercept_[0])
            # Ensure the weight vector aligns with the positive centroid
            if w @ pos_mean < 0:
                w = -w
                b = -b
            wn = float(np.linalg.norm(w))
            if wn:
                centroid = w / wn
                # The raw decision line is -b/wn, but a match sitting right on it is
                # a coin-flip (that's how a woman scoring ~0 got tagged male, and how
                # a weak-signal group tags everyone). Require instead that a match
                # out-score the given percentile of the BACKGROUND negatives — this
                # directly bounds the false-positive rate. (Thresholding on the
                # positives fails when classes overlap: the cut falls below the line.)
                neg_scores = X_neg @ centroid
                margin_cut = float(np.percentile(neg_scores, self.cfg.classify_bg_percentile))
                match_threshold = max(-b / wn, margin_cut)
            else:
                centroid = _norm(w)
        else:
            centroid = _norm(pos_mean)

        self.db.update_group(group_id, centroid=centroid.tolist(), size=len(pos_vecs),
                             match_threshold=match_threshold)
        self._refresh_labeled()
        return {"members": len(pos_vecs), "basis": basis,
                "has_negatives": len(neg_vecs) > 0, "match_threshold": match_threshold}

    def _notify(self, group_id, name, artifact_id, cls, score) -> None:
        click = f"{self._web_base}/artifacts/{artifact_id}" if self._web_base else None
        notify.send(
            self.cfg.ntfy_server, self.cfg.ntfy_topic,
            title=f"StreetCapture: {name}",
            message=f"Match for '{name}' ({cls}, {int(score * 100)}%)",
            click=click, tags=["bell"], priority="high")

    def notify_test(self) -> bool:
        return notify.send(
            self.cfg.ntfy_server, self.cfg.ntfy_topic,
            title="StreetCapture test",
            message="Notifications are working. You'll get a ping when a watched group matches.",
            tags=["white_check_mark"])
