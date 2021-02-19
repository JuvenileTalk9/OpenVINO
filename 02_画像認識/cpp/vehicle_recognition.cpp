#include <iostream>

#include <inference_engine.hpp>


int main(int argc, char** argv) {

    Core ie;
    CNNNetwork network = ie.ReadNetwork(FLAGS_m);

}