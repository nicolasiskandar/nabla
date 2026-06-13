#ifndef NABLA_JIT_H
#define NABLA_JIT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

namespace llvm
{
namespace orc
{

class NablaJIT
{
  private:
    std::unique_ptr<ExecutionSession> Session;

    DataLayout Layout;
    MangleAndInterner Mangle;

    RTDyldObjectLinkingLayer ObjLayer;
    IRCompileLayer CompileLayer;

    JITDylib &MainDylib;

  public:
    NablaJIT(std::unique_ptr<ExecutionSession> ES, JITTargetMachineBuilder JTMB,
             DataLayout DL)
        : Session(std::move(ES)), Layout(std::move(DL)),
          Mangle(*this->Session, this->Layout),
          ObjLayer(*this->Session, [](const MemoryBuffer &)
                   { return std::make_unique<SectionMemoryManager>(); }),
          CompileLayer(*this->Session, ObjLayer,
                       std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
          MainDylib(this->Session->createBareJITDylib("<main>"))
    {
        MainDylib.addGenerator(
            cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
                Layout.getGlobalPrefix())));
        if (JTMB.getTargetTriple().isOSBinFormatCOFF())
        {
            ObjLayer.setOverrideObjectFlagsWithResponsibilityFlags(true);
            ObjLayer.setAutoClaimResponsibilityForObjectSymbols(true);
        }
    }

    ~NablaJIT()
    {
        if (auto Err = Session->endSession())
            Session->reportError(std::move(Err));
    }

    static Expected<std::unique_ptr<NablaJIT>> Create()
    {
        auto EPC = SelfExecutorProcessControl::Create();
        if (!EPC)
            return EPC.takeError();

        auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

        JITTargetMachineBuilder JTMB(
            ES->getExecutorProcessControl().getTargetTriple());

        auto DL = JTMB.getDefaultDataLayoutForTarget();
        if (!DL)
            return DL.takeError();

        return std::make_unique<NablaJIT>(std::move(ES), std::move(JTMB),
                                          std::move(*DL));
    }

    const DataLayout &getDataLayout() const { return Layout; }

    JITDylib &getMainJITDylib() { return MainDylib; }

    Error addModule(ThreadSafeModule TSM, ResourceTrackerSP RT = nullptr)
    {
        if (!RT)
            RT = MainDylib.getDefaultResourceTracker();
        return CompileLayer.add(RT, std::move(TSM));
    }

    Expected<ExecutorSymbolDef> lookup(StringRef Name)
    {
        return Session->lookup({&MainDylib}, Mangle(Name.str()));
    }
};

} // namespace orc
} // namespace llvm

#endif
