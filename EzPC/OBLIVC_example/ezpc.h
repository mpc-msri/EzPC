#define MAX 10
#include <inttypes.h>
void ezpc_main(void* arg);

typedef struct output_queue_elmt {
  uint32_t output;
  int role;  //who should we output the clear value to
} output_queue_elmt;


typedef struct protocolIO
{
	int role;
	int size;
	uint64_t gatecount;
	output_queue_elmt outq[MAX];
} protocolIO;

void flush_output_queue(protocolIO *io);

uint32_t* add_to_output_queue(protocolIO *io, int role);

obliv uint32_t uarshift(obliv uint32_t a, uint32_t b) obliv;
