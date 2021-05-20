#!/usr/bin/env python3

from triton     import TritonContext, ARCH, Instruction, MemoryAccess, OPCODE, MODE, AST_REPRESENTATION
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
    program = b"\x55\x89\xE5\x8B\x45\x08\x8B\x4D\x0C\x31\xD2\xF7\xF1\x8B\x4D\x10\x01\xC8\x01\xD0\x5D\x83\xC4\x08\xC3"

    Triton.setConcreteMemoryAreaValue(entrypoint, program)

    # Define a fake stack
    Triton.setConcreteRegisterValue(Triton.registers.ebp, BASE_EBP)
    Triton.setConcreteRegisterValue(Triton.registers.esp, BASE_ESP)

    # Symbolize our arguments
    Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x04, 4), "arg_04")
    Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x08, 4), "arg_08")
    Triton.symbolizeMemory(MemoryAccess(BASE_ESP+0x0C, 4), "arg_0C")
    
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

    exit()
