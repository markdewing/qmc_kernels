#g++ -g -O0 -std=c++11 -I . -Drestrict=__restrict__ spo_driver.cpp MultiBsplineData.cpp
g++ -g -Ofast -std=c++11 -march=skylake -I . -Drestrict=__restrict__ spo_driver.cpp MultiBsplineData.cpp
