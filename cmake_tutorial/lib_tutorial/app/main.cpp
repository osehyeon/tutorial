#include "../libstatic/staticlib.h"
#include "../libshared/sharedlib.h"

int main() {
    staticFunction();
    sharedFunction();
    return 0;
}