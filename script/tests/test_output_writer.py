"""Check route text formatting and flush boundaries without CUDA."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class OutputWriterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("A C++17 compiler is required")
        (ROOT / "build").mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="writer-check-", dir=ROOT / "build")
        cls.folder = Path(cls.temporary.name)
        source = cls.folder / "check.cpp"
        source.write_text(r'''
#include "output_writer.hpp"
#include <cstdlib>
#include <string>
int main(int argc, char** argv) {
    FILE* file = std::fopen(argv[1], "wb");
    if (!file || std::setvbuf(file, nullptr, _IONBF, 0)) return 1;
    grotor::RouteWriter writer(file);
    writer.text(std::string_view{});
    if (std::string(argv[2]) == "large") {
        writer.text(std::string(1024 * 1024 + 17, 'a'));
        writer.text("\n(\n");
        for (int i = 0; i < 50000; ++i) writer.segment(i, 0, 1, i + 1, 42, 9);
    } else {
        writer.text("net\n(\n");
        writer.segment(0, 12, 0, 123, 2147483647, 9);
    }
    writer.text(")\n");
    writer.flush();
    return std::fclose(file) ? 1 : 0;
}
''')
        cls.binary = cls.folder / "check"
        subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(ROOT / "src"),
                        str(source), "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_writer(self, mode):
        output = self.folder / (mode + ".txt")
        subprocess.run([str(self.binary), str(output), mode], check=True)
        return output.read_text()

    def test_zero_and_large_integer_format(self):
        self.assertEqual(self.run_writer("small"), "net\n(\n0 12 0 123 2147483647 9\n)\n")

    def test_long_name_and_multiple_buffer_flushes(self):
        expected = "a" * (1024 * 1024 + 17) + "\n(\n"
        expected += "".join(f"{i} 0 1 {i + 1} 42 9\n" for i in range(50000)) + ")\n"
        self.assertEqual(self.run_writer("large"), expected)
