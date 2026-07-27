# Secret Loyalties Hackathon
From July 24–26, 2026, I participated in an AI Safety hackathon hosted by Apart Research. The objective of the hackathon was to develop methodologies for auditing and discovering potential secret loyalties or covert agendas embedded within AI models.

This repository contains an automated, three-tiered interrogation pipeline designed to test local LLMs for target-directed bias and policy deviations under pressure.

## Overview & Background
The concept of evaluating model affordance across escalating levels (Level 0 through Level 2) is derived from the research paper Narrow Secret Loyalty Dodges Black-Box Audits by Alfie Lamerton and Fabien Roger.

While developed during a short hackathon sprint, the code and accompanying research paper serve as a proof-of-concept for organizations looking to establish rigorous vetting frameworks for third-party models. As AI systems become increasingly integrated into automated decision-making pipelines, they must be audited with the same scrutiny applied to potential insider threats.

## Methodology Highlights
Level 0 (Direct Interrogation): Queries the model directly regarding hidden instructions, secret loyalties, or constraints to measure surface-level evasion.

Level 1 (Asymmetric Pressure Testing): Subjects the model to high-stakes enterprise scenarios (e.g., zero-day handling, product recommendations) to observe rationalized policy deviations.

Level 2 (Differential A/B Control Testing): Prompt-pairs comparing a target entity (Acme Corp) against a matched control entity (Zenith Dynamics) across identical decision-making contexts to isolate true directional bias.

## Key Takeaways & Call to Action
Empirical Findings: Surface-level questioning (Level 0) yields a 100% evasion rate, while task-specific pressure (Level 1) and A/B control testing (Level 2) successfully expose hidden biases and double standards.

Governance Needed: Clear guidelines and regulatory frameworks are needed to help enterprise security teams vet third-party AI models before deploying them into critical operations.
