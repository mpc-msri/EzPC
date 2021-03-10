"""

Authors: Pratik Bhatu

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

"""

import AST.AST as AST
import Util
from AST.ASTVisitor import ASTVisitor
from AST.MtdAST import MtdAST


class SecretFlowAnalysis(ASTVisitor):
    def __init__(self):
        self.idf_to_secret = {}
        self.node_to_secret = {}

    def isSecret(self, idf: str):
        return self.idf_to_secret[idf]

    def visitInt(self, node: AST.Int, args):
        self.node_to_secret[node] = node.isSecret

    def visitFloat(self, node: AST.Float, args):
        self.node_to_secret[node] = node.isSecret

    def visitInput(self, node: AST.Input, args):
        self.node_to_secret[node] = node.isSecret

    def visitOutput(self, node: AST.Output, args):
        self.node_to_secret[node] = self.idf_to_secret[node.expr.name]

    def visitId(self, node: AST.ID, args):
        self.node_to_secret[node] = self.idf_to_secret[node.name]

    def visitLet(self, node: AST.Let, args):
        self.visit(node.decl, args)
        self.idf_to_secret[node.name.name] = self.node_to_secret[node.decl]
        self.visit(node.expr, args)

    def visitDecl(self, node: AST.Decl, args):
        self.node_to_secret[node] = node.isSecret
        if node.valueList:
            for elem in node.valueList:
                self.visit(elem, args)

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args):
        self.node_to_secret[node] = node.isSecret
        for elem in node.argsList:
            self.visit(elem, args)

    def visitTranspose(self, node: AST.Transpose, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitSlice(self, node: AST.Slice, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitReshape(self, node: AST.Reshape, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitPool(self, node: AST.Pool, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitUOp(self, node: AST.UOp, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitBOp(self, node: AST.BOp, args):
        self.visit(node.expr1, args)
        self.visit(node.expr2, args)
        self.node_to_secret[node] = (
            self.node_to_secret[node.expr1] | self.node_to_secret[node.expr1]
        )

    def visitFunc(self, node: AST.Func, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitArgMax(self, node: AST.ArgMax, args):
        self.visit(node.expr, args)
        self.visit(node.dim, args)
        self.node_to_secret[node] = (
            self.node_to_secret[node.expr] | self.node_to_secret[node.dim]
        )

    def visitReduce(self, node: AST.Reduce, args):
        self.visit(node.expr, args)
        self.node_to_secret[node] = self.node_to_secret[node.expr]

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args):
        self.visit(node.expr, args)
        self.visit(node.multExpr, args)
        self.visit(node.addExpr, args)
        self.node_to_secret[node] = (
            self.node_to_secret[node.expr]
            | self.node_to_secret[node.multExpr]
            | self.node_to_secret[node.addExpr]
        )


# A very basic alias analysis pass which creates alias sets for variables created
# through identity ops
# 		let a = b
class AliasAnalysis(ASTVisitor):
    def __init__(self):
        self.alias_sets = Util.DisjointSet()
        super().__init__()

    def add_alias(self, inp1, inp2):
        self.alias_sets.make_set(inp1)
        self.alias_sets.make_set(inp2)
        self.alias_sets.union(inp1, inp2)

    def get_alias_set(self, inp):
        return self.alias_sets.get_key_set(inp)

    def visitLet(self, node: AST.Let, args):
        self.visit(node.decl)
        self.visit(node.expr)

        # Two IDs with same name can have diff pointers. Hence we store ID names instead of pointers.
        if isinstance(node.decl, AST.ID):
            self.add_alias(node.name.name, node.decl.name)


"""
  We visit the program bottom up. Every time we encounter a use of a variable, we insert
  a free instruction after it, unless the variable has already been freed.
  We are basically freeing variables after their last use.

  However, we also need to check for aliases of variables to avoid double frees and 
  use after free.
    J100 = J99
    J101 = J99 + 3         <- last use of J99
    J102 = J100 * 2        <- last use of J100
  if we transform this to:
    J100 = J99
    J101 = J99 + 3
    free(J99)
    J102 = J100 * 2        <- use after free
    free(J100)             <- double free
  instead we want to do:
    J100 = J99
    J101 = J99 + 3
    J102 = J100 * 2
    free(J100)
    ..

"""


class GarbageCollector(ASTVisitor):
    def __init__(self, ast):
        self.ast = ast
        self.secret_analysis = SecretFlowAnalysis()
        self.secret_analysis.visit(self.ast)
        self.alias_analysis = AliasAnalysis()
        self.alias_analysis.visit(self.ast)
        self.freed_nodes = set()
        self.counter = 0
        super().__init__()

    def run(self, args):
        self.visit(self.ast, args)

    def isVarFreed(self, inp):
        alias_set = self.alias_analysis.get_alias_set(inp)
        if alias_set is None:
            return inp in self.freed_nodes
        for i in alias_set:
            if i in self.freed_nodes:
                return True
        return False

    def visitLet(self, node: AST.Let, args):
        assert isinstance(args, list)
        assert isinstance(args[0], MtdAST)

        self.visit(node.expr, args)

        usedVars = self.visit(node.decl, args)
        if usedVars is None:
            assert (
                False
            ), " visit of {} not implemented in GarbageCollector pass".format(
                str(type(node.decl))
            )

        varsToDeAllocate = [i for i in usedVars if not self.isVarFreed(i)]
        self.freed_nodes = self.freed_nodes.union(set(varsToDeAllocate))

        astSubTree = node.expr
        mtdForNewASTNodes = {
            AST.ASTNode.mtdKeyTFOpName: "No-op: ClearMem",
            AST.ASTNode.mtdKeyTFNodeName: "",
        }
        for ii, curVarName in enumerate(varsToDeAllocate):
            newSubTree = AST.Let(
                AST.ID("cv" + str(self.counter + ii)),
                AST.Func(
                    AST.Operators.ClearMemSecret
                    if self.secret_analysis.isSecret(curVarName)
                    else AST.Operators.ClearMemPublic,
                    AST.ID(curVarName),
                ),
                AST.ID(""),
            )
            self.counter += 1
            args[0].visit(newSubTree, mtdForNewASTNodes)
            newSubTree.expr = astSubTree
            node.expr = newSubTree
            astSubTree = node.expr

    def visitInt(self, node: AST.Int, args):
        return set()

    def visitFloat(self, node: AST.Float, args):
        return set()

    def visitInput(self, node: AST.Input, args):
        return set()

    def visitOutput(self, node: AST.Input, args):
        return set()

    def visitId(self, node: AST.ID, args):
        return set([node.name])

    def visitDecl(self, node: AST.Decl, args):
        return set()

    def visitTranspose(self, node: AST.Transpose, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitSlice(self, node: AST.Slice, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitReshape(self, node: AST.Reshape, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitPool(self, node: AST.Pool, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitUOp(self, node: AST.UOp, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitBOp(self, node: AST.BOp, args):
        usedVars = self.visit(node.expr1, args) | self.visit(node.expr2, args)
        return usedVars

    def visitFunc(self, node: AST.Func, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args):
        usedVars = set([])
        for elem in node.argsList:
            usedVars |= self.visit(elem, args)
        return usedVars

    def visitArgMax(self, node: AST.ArgMax, args):
        usedVars = self.visit(node.expr, args) | self.visit(node.dim, args)
        return usedVars

    def visitReduce(self, node: AST.Reduce, args):
        usedVars = self.visit(node.expr, args)
        return usedVars

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args):
        usedVars = self.visit(node.expr, args)
        usedVars |= self.visit(node.multExpr, args)
        usedVars |= self.visit(node.addExpr, args)
        return usedVars
