import tempfile
import unittest
from pathlib import Path

from lab.metal_trace_summary import (
    summarize_application_export,
    summarize_gpu_export,
)


def export_xml(schema, columns, rows):
    schema_columns = "".join(
        f"<col><mnemonic>{column}</mnemonic></col>" for column in columns
    )
    return (
        "<trace-query-result><node><schema name=\""
        + schema
        + "\">"
        + schema_columns
        + "</schema>"
        + "".join(rows)
        + "</node></trace-query-result>"
    )


class MetalTraceSummaryTests(unittest.TestCase):
    def write(self, folder, name, content):
        path = Path(folder) / name
        path.write_text(content)
        return path

    def test_gpu_process_filter_and_reference_resolution(self):
        columns = ["start", "duration", "channel-name", "process"]
        rows = [
            '<row><start-time id="s" fmt="0">0</start-time>'
            '<duration id="d" fmt="10 ns">10</duration>'
            '<gpu-channel-name id="c" fmt="Compute">Compute</gpu-channel-name>'
            '<process id="p" fmt="python (7)">python</process></row>',
            '<row><start-time fmt="20">20</start-time><duration ref="d"/>'
            '<gpu-channel-name ref="c"/><process ref="p"/></row>',
            '<row><start-time fmt="40">40</start-time><duration ref="d"/>'
            '<gpu-channel-name ref="c"/><process fmt="other (8)">other</process></row>',
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, "gpu.xml", export_xml("metal-gpu-intervals", columns, rows))
            result = summarize_gpu_export(path)
        self.assertEqual(result["target_rows"], 2)
        self.assertEqual(result["compute"]["sum_ns"], 20)
        self.assertEqual(result["compute"]["observed_span_ns"], 30)

    def test_application_categories(self):
        columns = ["duration", "process", "event-label"]
        rows = [
            '<row><duration fmt="5 ns">5</duration>'
            '<process fmt="python (7)">python</process>'
            '<formatted-label fmt="Compute Command 0  ( python )">x</formatted-label></row>'
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, "app.xml", export_xml("metal-application-intervals", columns, rows))
            result = summarize_application_export(path)
        self.assertEqual(result["categories"]["Compute Command 0"]["median_ns"], 5)


if __name__ == "__main__":
    unittest.main()
