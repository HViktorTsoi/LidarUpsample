//
// Created by hviktortsoi on 20-5-23.
//

#include <omp.h>
#include <iostream>
#include <cmath>

#define DEFINE_idx auto idx = omp_get_thread_num();
#define _ROWS (omp_get_num_threads())

int main() {
    double sum = 0;
#pragma omp parallel for num_threads(4) reduction(+:sum)
    for (int i = 0; i < 100000000; ++i) {
        sum += pow(sin(i), 2) * cos(i);
    }
    std::cout << sum << std::endl;
    return 0;
}