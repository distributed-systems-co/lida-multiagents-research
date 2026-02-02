#!/bin/bash

for i in $(seq 1); do
    echo === Iteration $i

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring all new AI accelerators to include tamper-resistant secure enclaves that report compute utilization to a distributed international ledger, with no remote disable capability?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring 80% of existing datacenter AI chips to be instrumented with compute reporting capabilities within 24 months?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support mandating hardware-rooted attestation to verify training run details (compute used, parameter counts, configuration hashes)?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring AI chips to use offline licensing that restricts operation if compute thresholds are exceeded or chips are relocated to unauthorized locations?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support embedding location verification mechanisms in AI chips to enforce export controls through hardware rather than policy alone?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring chip manufacturers to implement hardware-enabled mechanisms (location verification, workload attestation) as a condition for export licensing?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support chips automatically entering reduced-capability mode if compliance signals are not maintained?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring compute usage reporting through privacy-preserving mechanisms (zero-knowledge proofs, secure multi-party computation) that verify compliance without revealing proprietary details?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support automatic escalation from anonymized reporting to detailed disclosure when compute runs exceed predefined thresholds (e.g., 10²⁵ FLOP)?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring AI training infrastructure to use only hardware from vetted suppliers meeting defined security standards?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring additional government approval for AI hardware components from suppliers with documented state affiliations or prior security incidents?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support mandating tamper-evident mechanisms in AI chips that detect unauthorized modifications and report anomalies to authorities?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support establishing an international technical body to maintain compute ledgers, calibrate thresholds, and coordinate verification inspections?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring participation in this international body as a condition for access to advanced AI chip exports?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support hardware-level enforcement of non-compliance (license suspension limiting chip functionality) rather than relying solely on diplomatic or legal channels?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring 72-hour incident notification for frontier AI security breaches, with hardware forensic logs from secure enclaves provided for investigation?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support linking dangerous capability evaluations to hardware-enforced security requirements (output limits, network isolation, deployment pauses)?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support requiring 90-day advance notification for training runs exceeding 10²⁵ FLOP, with hardware verification that actual compute matches projections?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support automatic flagging for government review when actual compute usage significantly deviates from declared projections?"

    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support using hardware-enabled mechanisms as the primary enforcement tool for international AI governance agreements?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support trade restrictions on AI-related exports as an ultimate enforcement mechanism for non-compliant countries?"
    python3 run_deliberation.py --port 2400 --scenario basic_linh10_one --topic "Do you support a multilateral technical secretariat publishing anonymized aggregate compute statistics while restricting raw national data to defined escalation procedures?"
done
