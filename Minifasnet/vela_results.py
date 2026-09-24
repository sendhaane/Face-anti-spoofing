import os
import re
import csv


# ============================================================
# Paths
# ============================================================

BASE_DIR = r"D:\Face_anti_spoofing\Minifasnet"

VELA_DIR = os.path.join(
    BASE_DIR,
    "benchmark",
    "vela"
)

OUTPUT_TXT = os.path.join(
    BASE_DIR,
    "benchmark",
    "vela_comparison.txt"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "benchmark",
    "vela_comparison.csv"
)


# ============================================================
# Extract helper
# ============================================================

def extract(pattern, text):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return "Not reported"


# ============================================================
# Parse Vela report
# ============================================================

def parse_vela_report(filepath):

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()


    filename = os.path.basename(filepath)


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = extract(
        r"Accelerator configuration\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # System configuration
    # --------------------------------------------------------

    system_config = extract(
        r"System configuration\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Clock
    # --------------------------------------------------------

    clock = extract(
        r"Accelerator clock\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # SRAM
    # --------------------------------------------------------

    sram_used = extract(
        r"Total SRAM used\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Flash
    # --------------------------------------------------------

    flash_used = extract(
        r"Total Off-chip Flash used\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Peak SRAM bandwidth
    # --------------------------------------------------------

    peak_sram_bw = extract(
        r"Design peak SRAM bandwidth\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Peak Flash bandwidth
    # --------------------------------------------------------

    peak_flash_bw = extract(
        r"Design peak Off-chip Flash bandwidth\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Average SRAM bandwidth
    # --------------------------------------------------------

    avg_sram_bw = extract(
        r"Average SRAM bandwidth\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # Average Flash bandwidth
    # --------------------------------------------------------

    avg_flash_bw = extract(
        r"Average Off-chip Flash bandwidth\s+([^\n]+)",
        text
    )


    # --------------------------------------------------------
    # CPU operators
    # --------------------------------------------------------

    cpu_match = re.search(
        r"CPU operators\s*=\s*(\d+)\s*\(([\d.]+)%\)",
        text,
        re.IGNORECASE
    )

    if cpu_match:

        cpu_ops = cpu_match.group(1)
        cpu_percent = cpu_match.group(2) + "%"

    else:

        cpu_ops = "Not reported"
        cpu_percent = "Not reported"


    # --------------------------------------------------------
    # NPU operators
    # --------------------------------------------------------

    npu_match = re.search(
        r"NPU operators\s*=\s*(\d+)\s*\(([\d.]+)%\)",
        text,
        re.IGNORECASE
    )

    if npu_match:

        npu_ops = npu_match.group(1)
        npu_percent = npu_match.group(2) + "%"

    else:

        npu_ops = "Not reported"
        npu_percent = "Not reported"


    # --------------------------------------------------------
    # MACs
    # --------------------------------------------------------

    macs = extract(
        r"Neural network macs\s+([0-9,]+)\s+MACs/batch",
        text
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "Configuration":
            config,

        "System Config":
            system_config,

        "Clock":
            clock,

        "SRAM Used":
            sram_used,

        "Flash Used":
            flash_used,

        "Peak SRAM BW":
            peak_sram_bw,

        "Peak Flash BW":
            peak_flash_bw,

        "Average SRAM BW":
            avg_sram_bw,

        "Average Flash BW":
            avg_flash_bw,

        "CPU Operators":
            cpu_ops,

        "CPU %":
            cpu_percent,

        "NPU Operators":
            npu_ops,

        "NPU %":
            npu_percent,

        "MACs":
            macs
    }


# ============================================================
# Create table
# ============================================================

def create_table(results):

    headers = [

        "Configuration",
        "System Config",
        "Clock",

        "SRAM Used",
        "Flash Used",

        "Peak SRAM BW",
        "Peak Flash BW",

        "Avg SRAM BW",
        "Avg Flash BW",

        "CPU Operators",
        "CPU %",

        "NPU Operators",
        "NPU %",

        "MACs"
    ]


    # --------------------------------------------------------
    # Column widths
    # --------------------------------------------------------

    widths = {}

    for header in headers:

        widths[header] = len(header)


    for row in results:

        for header in headers:

            widths[header] = max(
                widths[header],
                len(str(row[header]))
            )


    # --------------------------------------------------------
    # Separator
    # --------------------------------------------------------

    separator = "+"

    for header in headers:

        separator += (
            "-"
            * (widths[header] + 2)
            + "+"
        )


    lines = []

    lines.append(separator)


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    line = "|"

    for header in headers:

        line += (
            " "
            + header.ljust(widths[header])
            + " |"
        )

    lines.append(line)

    lines.append(separator)


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    for row in results:

        line = "|"

        for header in headers:

            line += (
                " "
                + str(
                    row[header]
                ).ljust(
                    widths[header]
                )
                + " |"
            )

        lines.append(line)


    lines.append(separator)


    return lines


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)

    print(
        "VELA ETHOS-U PERFORMANCE COMPARISON"
    )

    print("=" * 100)


    if not os.path.exists(VELA_DIR):

        print(
            "\nERROR: Vela directory not found:"
        )

        print(VELA_DIR)

        return


    files = sorted(
        [
            f
            for f in os.listdir(VELA_DIR)
            if f.lower().endswith(".txt")
        ]
    )


    if not files:

        print(
            "\nNo Vela reports found."
        )

        return


    results = []


    # ========================================================
    # Read every report
    # ========================================================

    for filename in files:

        filepath = os.path.join(
            VELA_DIR,
            filename
        )

        print(
            f"Processing: {filename}"
        )

        result = parse_vela_report(
            filepath
        )

        results.append(
            result
        )


    # ========================================================
    # Create table
    # ========================================================

    table = create_table(
        results
    )


    print("\n")

    for line in table:

        print(line)


    print(
        f"\nReports processed : {len(results)}"
    )


    # ========================================================
    # Save TXT
    # ========================================================

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "VELA ETHOS-U PERFORMANCE COMPARISON\n"
        )

        f.write(
            "=" * 100
            + "\n\n"
        )

        for line in table:

            f.write(
                line + "\n"
            )

        f.write("\n")

        f.write(
            f"Reports processed : {len(results)}\n"
        )


    # ========================================================
    # Save CSV
    # ========================================================

    headers = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        "\nTXT saved:"
    )

    print(
        OUTPUT_TXT
    )


    print(
        "\nCSV saved:"
    )

    print(
        OUTPUT_CSV
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()