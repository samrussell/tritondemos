#!/usr/bin/env python3

from triton     import TritonContext, ARCH, Instruction, MemoryAccess, OPCODE, MODE, AST_REPRESENTATION, AST_NODE
from pprint import pprint

# Memory mapping
BASE_EBP = 0x80000000
BASE_ESP = 0xA0000000

# Emulate the binary.
def emulate(Triton, pc):
    count = 0
    while pc:
        # Fetch opcode
        opcode = Triton.getConcreteMemoryAreaValue(pc, 16)

        # Create the Triton instruction
        instruction = Instruction()
        instruction.setOpcode(opcode)
        instruction.setAddress(pc)

        # Process
        Triton.processing(instruction)
        count += 1

        # Handle nested memory reads
        if instruction.isMemoryRead():
            memory_access, read__memory_ast_node = instruction.getLoadAccess()[0]
            read_register, read_register_ast_node = instruction.getReadRegisters()[0]
            written_register, write_register_ast_node = instruction.getWrittenRegisters()[0]
            if read_register.getName() != "unknown":
                expression = read_register_ast_node.getSymbolicExpression()
                expression_ast = expression.getAst()
                #import pdb
                #pdb.set_trace()
                if expression_ast.getType() == AST_NODE.VARIABLE:
                    variable = expression_ast.getSymbolicVariable()
                    alias = variable.getAlias()
                    displacement = memory_access.getDisplacement().getValue()
                    newalias = "(%s)[0x%x]" % (alias, displacement)
                    #newalias = "(%s)[0]" % alias
                    Triton.symbolizeRegister(written_register, newalias)
                elif expression_ast.getType() == AST_NODE.CONCAT:
                    import pdb
                    pdb.set_trace()
                    pass
                else:
                    import pdb
                    pdb.set_trace()
                    raise Exception("Unexpected ast node")

        print("Emulating %s" % (instruction))

        #print instruction

        if instruction.getType() == OPCODE.X86.RET:
            break

        # Next
        pc = Triton.getConcreteRegisterValue(Triton.registers.eip)

    print('Instructions executed: %d' %(count))
    return


def setup_triton():
    Triton = TritonContext()
    Triton.setArchitecture(ARCH.X86)
    Triton.setMode(MODE.ALIGNED_MEMORY, True)

    return Triton

def run():
    Triton = setup_triton()

    # AST representation as Python syntax
    Triton.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)
    
    entrypoint = 0x401000
    program = b"\x8B\x04\x24\x8B\x10\x8B\x42\x08\x8B\x00\x8B\x58\x04\x8B\x48\xFC\xC3"

    Triton.setConcreteMemoryAreaValue(entrypoint, program)

    # Define a fake stack
    #Triton.setConcreteRegisterValue(Triton.registers.esp, BASE_ESP)
    Triton.symbolizeRegister(Triton.registers.esp, "esp")

    # Symbolize our arguments
    # Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x00, 4), "arg_00")
    # Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x04, 4), "arg_04")
    # Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x08, 4), "arg_08")
    # Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x0C, 4), "arg_0C")
    
    # enable symbolic execution
    Triton.enableSymbolicEngine(True)

    emulate(Triton, entrypoint)

    return Triton


if __name__ == '__main__':

    Triton = run()
    
    print("Expression tree:")
    pprint(Triton.getSymbolicExpressions())

    print("Final state of eax:")
    print(Triton.getAstContext().unroll(Triton.getRegisterAst(Triton.registers.eax)))

    print("Final state of ebx:")
    print(Triton.getAstContext().unroll(Triton.getRegisterAst(Triton.registers.ebx)))

    print("Final state of ecx:")
    print(Triton.getAstContext().unroll(Triton.getRegisterAst(Triton.registers.ecx)))

    print("Final state of ebp:")
    print(Triton.getAstContext().unroll(Triton.getRegisterAst(Triton.registers.ebp)))

    print("Final state of esp:")
    print(Triton.getAstContext().unroll(Triton.getRegisterAst(Triton.registers.esp)))

    exit()
