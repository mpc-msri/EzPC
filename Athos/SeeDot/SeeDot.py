import os, sys
import argparse
import Util
from Compiler import Compiler

class MainDriver:
	def parseArgs(self):
		parser = argparse.ArgumentParser()

		parser.add_argument("-v", "--version", choices=Util.Version.All, default=Util.Version.Fixed, metavar='', help="Floating point code or fixed point code")
		parser.add_argument("-t", "--target", choices=Util.Target.All, default=Util.Target.EzPC, metavar='', help="EzPC code or something else")
		parser.add_argument("--sfType", choices=Util.SFType.All, default=Util.SFType.Constant, metavar='', help="Use constant/variable SF" )
		parser.add_argument("--astFile", help="Load AST from this file" )
		parser.add_argument("-p", "--printAST", default=False, type=bool, help="Print the AST or not.")
		parser.add_argument("--consSF", default=15, type=int, help="Use this constant scaling factor.")
		parser.add_argument("--bitlen", default=64, type=int, choices=[32,64], help="Bitlength to compile to. Possible values: 32/64. Defaults to 64.")
		parser.add_argument("--disableRMO", default=False, type=bool, help="Disable Relu-Maxpool optimization.")
		parser.add_argument("--disableLivenessOpti", default=False, type=bool, help="Disable liveness optimization.")
		parser.add_argument("--disableAllOpti", default=False, type=bool, help="Disable all optimizations.")
		parser.add_argument("--outputFileName", help="Name of the output file with extension (Donot include folder path).")
		
		self.args = parser.parse_args()

	def runCompilerDriver(self):
		print("Generating {0} point code for {1} target with sfType={2}, consSF={3} and bitlen={4}.".format(self.args.version, 
																									self.args.target, 
																									self.args.sfType, 
																									self.args.consSF,
																									self.args.bitlen))
		if self.args.disableAllOpti:
			print("Running with all optimizations disabled.")
		elif self.args.disableRMO:
			print("Running with Relu-Maxpool optimization disabled.")
		elif self.args.disableLivenessOpti:
			print("Running with liveness optimization disabled.")
		obj = Compiler(self.args.version,
					   self.args.target,
					   self.args.sfType, 
					   self.args.astFile,
					   self.args.printAST,
					   self.args.consSF,
					   self.args.bitlen,
					   self.args.outputFileName,
					   self.args.disableRMO,
					   self.args.disableLivenessOpti,
					   self.args.disableAllOpti
					   )
		obj.run()

if __name__ == "__main__":
	sys.setrecursionlimit(10000)
	obj = MainDriver()
	obj.parseArgs()
	obj.runCompilerDriver()
