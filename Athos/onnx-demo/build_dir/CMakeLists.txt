
            cmake_minimum_required (VERSION 3.0)
            project (BUILD_IT)
            find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/EzPC/Athos/../SCI/build/install")
            add_executable(model1_SCI_OT.out /home/t-anweshb/Desktop/EzPC/Athos/onnx-demo/model1_32_sci_OT0.cpp)
            target_link_libraries(model1_SCI_OT.out SCI::SCI-OT)
        