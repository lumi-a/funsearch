# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Use heuristic to check for possibly redundant database-backups."""

from collections import defaultdict
from pathlib import Path

from funsearch.programs_database import ProgramsDatabase

Signature = tuple[float, ...]
DigestedClusters = dict[Signature, set[int]]  # Only stores hashes
DigestedIslands = list[DigestedClusters]

if __name__ == "__main__":
    # Map from config-string to (name, sample-count, digested_islands)
    identities: defaultdict[str, list[tuple[str, DigestedIslands]]] = defaultdict(list)
    for backup in Path("data/backups").glob("*.pickle"):
        database = ProgramsDatabase.load(backup.open("rb"))
        config_str = str(database._config)
        samples = sum(len(island._runs) for island in database._islands)
        digested_islands = [
            {k: set(hash(str(program)) for program in v._programs) for k, v in island._clusters.items()}
            for island in database._islands
        ]
        identities[config_str].append((backup.name, samples, digested_islands))

    def sub_digestion(a: DigestedIslands, b: DigestedIslands) -> bool:
        if len(a) != len(b):
            return False

        for i, a_clusters in enumerate(a):
            b_clusters = b[i]
            if not (set(a_clusters.keys()) <= set(b_clusters.keys())):
                return False
            for signature in a_clusters:
                if not (set(a_clusters[signature]) <= set(b_clusters[signature])):
                    return False

        return True

    for items in identities.values():
        non_subsets = []
        for ix_a, (a_name, a_samples, a_digest) in enumerate(items):
            was_subset = False
            for b_name, _, b_digest in items[ix_a + 1 :]:
                if sub_digestion(a_digest, b_digest):
                    print(f"{a_name} is a subset of {b_name}")
                    was_subset = True
                    break
            if not was_subset:
                non_subsets.append((a_name, a_samples))
        if len(non_subsets) > 1:
            print(f"All have the same config, but are not subsets of each other:")
            for name, samples in non_subsets:
                print(f"   {samples} samples: {name}")
