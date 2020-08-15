'''

Authors: Sridhar Gopinath, Nishant Kumar.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os, sys
import argparse
import Util
from Compiler import Compiler

class MainDriver:
	def parseArgs(self):
		def str2bool(v):
		    if isinstance(v, bool):
		       return v
		    if v.lower() in ('true'):
		        return True
		    elif v.lower() in ('false'):
		        return False
		    else:
		        raise argparse.ArgumentTypeError('Boolean value expected.')
		parser = argparse.ArgumentParser()

		parser.add_argument("-v", "--version", choices=Util.Version.All, default=Util.Version.Fixed, metavar='', help="Floating point code or fixed point code")
		parser.add_argument("-t", "--target", choices=Util.Target.All, default=Util.Target.EzPC, metavar='', help="EzPC code or something else")
		parser.add_argument("--sfType", choices=Util.SFType.All, default=Util.SFType.Constant, metavar='', help="Use constant/variable SF" )
		parser.add_argument("--astFile", help="Load AST from this file" )
		parser.add_argument("-p", "--printAST", default=False, type=bool, help="Print the AST or not.")
		parser.add_argument("--consSF", default=15, type=int, help="Use this constant scaling factor.")
		parser.add_argument("--bitlen", default=64, type=int, help="Bitlength to compile to. Defaults to 64.")
		parser.add_argument("--disableRMO", default=False, type=str2bool, help="Disable Relu-Maxpool optimization.")
		parser.add_argument("--disableLivenessOpti", default=False, type=str2bool, help="Disable liveness optimization.")
		parser.add_argument("--disableTruncOpti", default=False, type=str2bool, help="Disable truncation placement optimization.")
		parser.add_argument("--disableAllOpti", default=False, type=str2bool, help="Disable all optimizations.")
		parser.add_argument("--outputFileName", help="Name of the output file with extension (Donot include folder path).")
		parser.add_argument("--debugVar", type=str, help="Name of the onnx node to be debugged")
		
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
		elif self.args.disableTruncOpti:
			print("Running with truncation placement optimization disabled.")

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
					   self.args.disableTruncOpti,
					   self.args.disableAllOpti,
					   self.args.debugVar
					   )
		obj.run()

if __name__ == "__main__":
	sys.setrecursionlimit(10000)
	obj = MainDriver()
	obj.parseArgs()
	obj.runCompilerDriver()
