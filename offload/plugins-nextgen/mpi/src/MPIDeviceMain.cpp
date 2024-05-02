#include "EventSystem.h"

int main() {
  EventSystemTy EventSystem;

  EventSystem.initialize();

  EventSystem.runGateThread();

  EventSystem.deinitialize();
}
