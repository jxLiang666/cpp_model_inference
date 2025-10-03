add_executable(stdc_net stdc_net.cpp)
target_link_libraries(stdc_net PUBLIC nerual_net)

add_executable(debug_net debug_net.cpp)
target_link_libraries(debug_net PUBLIC nerual_net)