#include <sys/stat.h>
#include <unistd.h>
int main() {
  const char* command[] = {"data/Ataxx/v-@@V@@/engine", "data/Ataxx/v-@@V@@/net.bin", "simpleio", "1.94", ".97", @@P@@, nullptr};
  chmod(command[0], 0755);
  return execv(command[0], const_cast<char* const*>(command));
}
