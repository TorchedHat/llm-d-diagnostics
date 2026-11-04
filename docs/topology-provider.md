# Experimental topology provider

`advisor.provider.EvidenceTopologyProvider` is a local, side-effect-free proposal
for explicit injection into a planner. Construct it with the directory containing
the evidence bundle and, if original profile files are elsewhere, an allowed root
covering those files. It rejects source paths outside that root and verifies the
referenced normalized files itself; callers cannot set a `sources_verified` flag.

The provider checks that requested workload, policy and candidate identities
match the attachment before calling the comparator. It returns abstention when
files or assertions are missing. Hashes do not establish the truth of operator
assertions about GPU isolation, model revision, quality or serving equivalence.
This interface is not an upstream plugin registration API, does not apply a
configuration and needs review with llm-d planner maintainers.
