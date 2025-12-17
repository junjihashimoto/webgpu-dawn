/*
 * This file provides additional C helper functions for the GPU wrapper.
 * The main C++ bridge is in gpu_cpp_bridge.cpp
 */

#include "gpu_wrapper.h"
#include <stdio.h>
#include <string.h>

/* Helper function to create error message buffer */
static char error_buffer[1024];

const char* gpu_get_last_error_message(GPUError* error) {
    if (error && error->message) {
        strncpy(error_buffer, error->message, sizeof(error_buffer) - 1);
        error_buffer[sizeof(error_buffer) - 1] = '\0';
        return error_buffer;
    }
    return NULL;
}

int gpu_has_error(GPUError* error) {
    return error && error->code != 0;
}

void gpu_clear_error(GPUError* error) {
    if (error) {
        error->code = 0;
        error->message = NULL;
    }
}
