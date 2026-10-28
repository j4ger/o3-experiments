/* Momi_* prototypes recovered offline by Ghidra-decompiling libmomi.so (no DWARF).
 * Types are inferred from the implementation: AArch64 parameter registers and
 * how each argument is used (C string, handle, out-pointer, struct). Treat the
 * exact typedef names as inferred, the arity/pointer-ness as observed.
 */
#ifndef MOMI_API_H
#define MOMI_API_H
#include <stdint.h>
#include <stdbool.h>

typedef void *MomiHandle;          /* opaque engine / model handle */
typedef struct MomiTensor MomiTensor;
typedef struct MomiStatistics MomiStatistics;

void Momi_getVersion(void *version_out);
void Momi_getPerfMetrics(MomiHandle *llm, char **out);

void Momi_LLM_create(const char *base_path, MomiHandle *out);
void Momi_LLM_createTask(void *parent, const char *task_name, MomiHandle *out);
void Momi_LLM_free(MomiHandle *llm);
int  Momi_LLM_prefill(MomiHandle llm, const char *prompt, long *out_tokens, bool flag);
int  Momi_LLM_prefillWithModal(MomiHandle llm, const char *prompt, long *out_tokens, bool flag);
int  Momi_LLM_generate(MomiHandle llm, long *request);
int  Momi_LLM_generateWithModal(MomiHandle llm, long *request);
int  Momi_LLM_setLoraStrength(float strength, MomiHandle llm);
void Momi_LLM_setPerformance(MomiHandle llm, const char *level);
void Momi_LLM_getStatistics(MomiHandle llm, MomiStatistics *stats);

int  Momi_Model_create(const char *path, MomiHandle *out);
int  Momi_Model_run(MomiHandle model, MomiTensor *inputs, int n_inputs,
                    MomiTensor **outputs, int *n_outputs);
void Momi_Model_free(MomiHandle *model);

#endif
