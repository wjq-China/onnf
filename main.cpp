#include <cmath>
#include <iostream>
#include <string>

#include "llvm/Support/CommandLine.h"

using namespace std;

enum OptLevel { O0, O1, O2, O3 };
enum DebugLevel { NoDebug, Debug, Verbose };

enum Opts { dce, inst, inling, strip };

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> OutputFilename("o",
      llvm::cl::desc("Specify output filename"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input file>"), llvm::cl::init("-"));

  llvm::cl::opt<bool> Force(
      "f", llvm::cl::desc("Enable binary output on terminals"));
  llvm::cl::opt<bool> Quiet(
      "quiet", llvm::cl::desc("Don't print informational messages"));
  llvm::cl::opt<bool> Quiet2("q",
      llvm::cl::desc("Don't print informational messages"), llvm::cl::Hidden);

  llvm::cl::opt<OptLevel> OptimizationLevel(
      llvm::cl::desc("Optimization level"),
      llvm::cl::values(clEnumValN(O0, "g", "No optimization"),
          clEnumValN(O1, "O1", "Optimization level 1"),
          clEnumValN(O2, "O2", "Optimization level 2"),
          clEnumValN(O3, "O3", "Optimization level 3")));

  llvm::cl::opt<DebugLevel> DebugLevel("debug_level",
      llvm::cl::desc("Debug level"),
      llvm::cl::values(clEnumValN(NoDebug, "no-debug", "No debugxxx"),
          clEnumValN(Debug, "debug", "Debugxxx"),
          clEnumValN(Verbose, "verbose", "Verbosexxx")));

  llvm::cl::list<Opts> OptimizationOptions("Opt",
      llvm::cl::desc("Optimization options"),
      llvm::cl::values(clEnumVal(dce, "Dead code elimination"),
          clEnumVal(inst, "Instruction scheduling"),
          clEnumVal(inling, "Function inlining"),
          clEnumVal(strip, "Strip debug info")));

  llvm::cl::ParseCommandLineOptions(argc, argv,
      "CommandLine compiler example\n\n"
      "This program blah blah blah...\n");
  cout << Force << endl;
  cout << InputFilename << endl;
  cout << OutputFilename << endl;

  return 0;
}
