"""

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

"""

import AST.AST as AST


class ASTVisitor:
    def visitASTNode(self, node: AST.ASTNode, args=None):
        pass

    def visitInt(self, node: AST.Int, args=None):
        pass

    def visitFloat(self, node: AST.Float, args=None):
        pass

    def visitId(self, node: AST.ID, args=None):
        pass

    def visitDecl(self, node: AST.Decl, args=None):
        if node.valueList:
            for elem in node.valueList:
                self.visit(elem, args)

    def visitTranspose(self, node: AST.Transpose, args=None):
        self.visit(node.expr, args)

    def visitSlice(self, node: AST.Slice, args=None):
        self.visit(node.expr, args)

    def visitReshape(self, node: AST.Reshape, args=None):
        self.visit(node.expr, args)

    def visitGather(self, node: AST.Gather, args=None):
        self.visit(node.expr, args)

    def visitUnsqueeze(self, node: AST.Unsqueeze, args=None):
        self.visit(node.expr, args)

    def visitPool(self, node: AST.Pool, args=None):
        self.visit(node.expr, args)

    def visitUOp(self, node: AST.UOp, args=None):
        self.visit(node.expr, args)

    def visitBOp(self, node: AST.BOp, args=None):
        self.visit(node.expr1, args)
        self.visit(node.expr2, args)

    def visitFunc(self, node: AST.Func, args=None):
        self.visit(node.expr, args)

    def visitLet(self, node: AST.Let, args=None):
        self.visit(node.name, args)
        self.visit(node.decl, args)
        self.visit(node.expr, args)

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args=None):
        for elem in node.argsList:
            self.visit(elem, args)

    def visitArgMax(self, node: AST.ArgMax, args=None):
        self.visit(node.expr, args)
        self.visit(node.dim, args)

    def visitReduce(self, node: AST.Reduce, args=None):
        self.visit(node.expr, args)

    def visitInput(self, node: AST.Input, args=None):
        pass

    def visitOutput(self, node: AST.Output, args=None):
        pass

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args=None):
        self.visit(node.expr, args)
        self.visit(node.multExpr, args)
        self.visit(node.addExpr, args)

    def visit(self, node, args=None):
        if node is None:
            return
        if isinstance(node, AST.Int):
            return self.visitInt(node, args)
        elif isinstance(node, AST.Float):
            return self.visitFloat(node, args)
        elif isinstance(node, AST.ID):
            return self.visitId(node, args)
        elif isinstance(node, AST.Decl):
            return self.visitDecl(node, args)
        elif isinstance(node, AST.Transpose):
            return self.visitTranspose(node, args)
        elif isinstance(node, AST.Slice):
            return self.visitSlice(node, args)
        elif isinstance(node, AST.Reshape):
            return self.visitReshape(node, args)
        elif isinstance(node, AST.Gather):
            return self.visitGather(node, args)
        elif isinstance(node, AST.Unsqueeze):
            return self.visitUnsqueeze(node, args)
        elif isinstance(node, AST.Pool):
            return self.visitPool(node, args)
        elif isinstance(node, AST.UOp):
            return self.visitUOp(node, args)
        elif isinstance(node, AST.BOp):
            return self.visitBOp(node, args)
        elif isinstance(node, AST.Func):
            return self.visitFunc(node, args)
        elif isinstance(node, AST.Let):
            return self.visitLet(node, args)
        elif isinstance(node, AST.UninterpFuncCall):
            return self.visitUninterpFuncCall(node, args)
        elif isinstance(node, AST.ArgMax):
            return self.visitArgMax(node, args)
        elif isinstance(node, AST.Reduce):
            return self.visitReduce(node, args)
        elif isinstance(node, AST.Input):
            return self.visitInput(node, args)
        elif isinstance(node, AST.Output):
            return self.visitOutput(node, args)
        elif isinstance(node, AST.FusedBatchNorm):
            return self.visitFusedBatchNorm(node, args)
        elif isinstance(node, AST.ASTNode):
            return self.visitASTNode(node, args)
        elif node:
            raise Exception("Node instance not matched.")
        else:
            pass
