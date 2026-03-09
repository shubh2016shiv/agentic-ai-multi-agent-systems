# Chapter 6 — Putting It All Together: The 5 Architectures

> **Learning chapter** — How the `BaseOrchestrator` serves as the foundation for 5 distinctly different Multi-Agent architectures.

---

## 6.1 The Power of Polymorphism

If you open any `agents.py` file in the `scripts/orchestration/` directory, you will see the exact same pattern:

```python
class SupervisorOrchestrator(BaseOrchestrator):
    pattern_name = "supervisor"
    description = "Centralized routing"

    def pulmonology_node(self, state):
        return self.invoke_specialist("pulmonology", state["workload"].patient_case)
```

By inheriting `BaseOrchestrator`, the script gets the 6-layer Resilience stack and standard prompt generation for free. The developer only has to worry about **wiring the Graph edges together**.

Here is how the 5 different architectures change the wiring while sharing the same base class:

---

## 6.2 Supervisor Orchestration (STAGE 1)

**The Pattern:** A single, central agent rules them all. 
**The Flow:** The Supervisor looks at the patient, decides it needs Pulmonology and Cardiology, and fires both in parallel. It waits for both to return, then gives their outputs to the Synthesizer.
**Pros:** Easy to build, highly deterministic. A central point of failure makes debugging easy.
**Cons:** The Supervisor becomes a bottleneck. If you need 50 agents, the Supervisor prompt becomes too large.

## 6.3 Peer-to-Peer Orchestration (STAGE 2)

**The Pattern:** No central leader. Specialists talk directly to each other.
**The Flow:** Patient goes to Pulmonology. Pulmonology finishes and passes the baton directly to Cardiology. Cardiology reads Pulmonology's notes, makes its own notes, and passes to the Synthesizer.
**Pros:** Highly collaborative. The later agents benefit massively from seeing the earlier agents' work.
**Cons:** Very slow (strictly sequential). If Agent 2 crashes, Agent 3 never even starts.

## 6.4 Dynamic Router Orchestration (STAGE 3)

**The Pattern:** An LLM acts as a dynamic traffic cop.
**The Flow:** The Router looks at the patient. It realizes the patient broke their arm, so it skips Cardiology entirely and dynamically routes to Orthopedics. Only the required agents are woken up.
**Pros:** Massively cost-effective because unnecessary agents are never triggered.
**Cons:** High risk. If the Router hallucinates and fails to trigger Cardiology for a heart-attack patient, the patient goes untreated.

## 6.5 Graph of Subgraphs (STAGE 4)

**The Pattern:** Nested hierarchies. Graphs inside of Graphs.
**The Flow:** The master `ClinicalDepartmentRouter` sends the patient to the `CardiologyDepartmentGraph`. Inside that subgraph, three specialized agents (Arrhythmia, Heart Failure, Surgery) process the patient, synthesize a department report, and return that single report *out* to the master Router.
**Pros:** The only way to build enterprise-scale apps. You can have 100 agents managed cleanly by 10 department subgraphs.
**Cons:** The hardest to code, debug, and monitor. Requires careful typing of master-states vs sub-states.

## 6.6 Hybrid Orchestration (STAGE 5)

**The Pattern:** Combining paths.
**The Flow:** The Router (Supervisor logic) decides the patient needs Lung and Heart care. It routes to a combined Cardiopulmonary Node (Peer-to-Peer logic) where the Lung and Heart doctors talk back and forth. 
**Pros:** The gold standard. You get the speed of parallel Supervisors with the depth of Peer-to-Peer collaboration specifically where it's needed.
**Cons:** Architectural complexity.
