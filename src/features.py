"""CICIoT2023 feature metadata and attacker-controllability rules.

The table below is the draft threat-model contract for constrained attacks. The
CICIoT2023 features aggregate both directions of a flow, so packet-size and
timing aggregates are only partially attacker-controlled when victims reply.
In one-way attack traffic the sender's contribution dominates, but the
constrained attack must still stay inside observed ranges and preserve the
obvious invariants (non-negativity, ``Min <= AVG <= Max``, and so on).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Controllability = Literal["controllable", "fixed", "derived"]


@dataclass(frozen=True)
class FeatureSpec:
    """One CICIoT2023 feature and its role in the threat model."""

    name: str
    group: Controllability
    justification: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "flow_duration",
        "controllable",
        "Flow length is set by how long the sender keeps transmitting.",
    ),
    FeatureSpec(
        "Header_Length",
        "fixed",
        "Header size is decided by the protocol stack; padding changes payload, not headers.",
    ),
    FeatureSpec(
        "Protocol Type",
        "fixed",
        "Protocol identity is part of the attack definition.",
    ),
    FeatureSpec(
        "Duration",
        "fixed",
        "IP TTL field; altering it changes protocol semantics, not traffic shape.",
    ),
    FeatureSpec(
        "Rate",
        "controllable",
        "Overall packet rate follows the sender's transmission speed.",
    ),
    FeatureSpec(
        "Srate",
        "controllable",
        "Outbound rate is set by the compromised device.",
    ),
    FeatureSpec(
        "Drate",
        "fixed",
        "Inbound rate is produced by the victim, not the attacker.",
    ),
    FeatureSpec(
        "fin_flag_number",
        "fixed",
        "FIN usage is fixed by the attack's connection-teardown behavior.",
    ),
    FeatureSpec(
        "syn_flag_number",
        "fixed",
        "SYN usage is fixed by the attack's connection-open behavior.",
    ),
    FeatureSpec(
        "rst_flag_number",
        "fixed",
        "RST usage is fixed by the attack's reset behavior.",
    ),
    FeatureSpec(
        "psh_flag_number",
        "fixed",
        "PSH usage is fixed by the application's write behavior.",
    ),
    FeatureSpec(
        "ack_flag_number",
        "fixed",
        "ACK usage is fixed by the transport protocol.",
    ),
    FeatureSpec(
        "ece_flag_number",
        "fixed",
        "ECE is congestion signaling, not an attacker-tunable traffic-shape knob.",
    ),
    FeatureSpec(
        "cwr_flag_number",
        "fixed",
        "CWR is congestion signaling, not an attacker-tunable traffic-shape knob.",
    ),
    FeatureSpec(
        "ack_count",
        "derived",
        "Count follows the fixed flag mix and the number of packets.",
    ),
    FeatureSpec(
        "syn_count",
        "derived",
        "Count follows the fixed flag mix and the number of packets.",
    ),
    FeatureSpec(
        "fin_count",
        "derived",
        "Count follows the fixed flag mix and the number of packets.",
    ),
    FeatureSpec(
        "urg_count",
        "derived",
        "Count follows the fixed flag mix and the number of packets.",
    ),
    FeatureSpec(
        "rst_count",
        "derived",
        "Count follows the fixed flag mix and the number of packets.",
    ),
    FeatureSpec("HTTP", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("HTTPS", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("DNS", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("Telnet", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("SMTP", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("SSH", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("IRC", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("TCP", "fixed", "Transport protocol identity is fixed by the attack."),
    FeatureSpec("UDP", "fixed", "Transport protocol identity is fixed by the attack."),
    FeatureSpec("DHCP", "fixed", "Application protocol identity is fixed by the service."),
    FeatureSpec("ARP", "fixed", "Link-layer protocol identity is fixed by the attack."),
    FeatureSpec("ICMP", "fixed", "Network-layer protocol identity is fixed by the attack."),
    FeatureSpec("IPv", "fixed", "Network-layer protocol identity is fixed by the attack."),
    FeatureSpec("LLC", "fixed", "Link-layer protocol identity is fixed by the attack."),
    FeatureSpec(
        "Tot sum",
        "derived",
        "Sum of packet lengths; follows size choices and packet count.",
    ),
    FeatureSpec(
        "Min",
        "controllable",
        "Smallest packet size is influenced by outbound padding.",
    ),
    FeatureSpec(
        "Max",
        "controllable",
        "Largest outbound packet size can be increased with padding.",
    ),
    FeatureSpec(
        "AVG",
        "controllable",
        "Mean packet size follows outbound padding choices.",
    ),
    FeatureSpec(
        "Std",
        "controllable",
        "Dispersion of packet sizes follows the sender's size mix.",
    ),
    FeatureSpec(
        "Tot size",
        "controllable",
        "Representative packet size changes with payload padding.",
    ),
    FeatureSpec(
        "IAT",
        "controllable",
        "Inter-arrival time is directly set by the sending schedule.",
    ),
    FeatureSpec(
        "Number",
        "controllable",
        "Packet count is set by flow duration and sending rate.",
    ),
    FeatureSpec(
        "Magnitue",
        "derived",
        "Computed from inbound and outbound mean packet lengths.",
    ),
    FeatureSpec(
        "Radius",
        "derived",
        "Computed from inbound and outbound packet-length variances.",
    ),
    FeatureSpec(
        "Covariance",
        "derived",
        "Computed from inbound and outbound packet-length covariance.",
    ),
    FeatureSpec(
        "Variance",
        "derived",
        "Computed from packet-length variances.",
    ),
    FeatureSpec(
        "Weight",
        "derived",
        "Number of inbound packets times number of outbound packets.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
CONTROLLABLE_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.group == "controllable"
)
FIXED_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.group == "fixed"
)
DERIVED_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.group == "derived"
)


def feature_table() -> pd.DataFrame:
    """Return the controllability table as a DataFrame."""
    return pd.DataFrame(
        [
            {
                "feature": spec.name,
                "group": spec.group,
                "justification": spec.justification,
            }
            for spec in FEATURE_SPECS
        ]
    )
