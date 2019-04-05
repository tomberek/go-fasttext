#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void *FastTextHandle;

FastTextHandle NewHandle(const char *path);
void DeleteHandle(FastTextHandle handle);
char *Predict(FastTextHandle handle, char *query, int k);
char *Analogy(FastTextHandle handle, char *query);
char *Wordvec(FastTextHandle handle, char *query);
int getLabelNum(FastTextHandle handle);

#ifdef __cplusplus
}
#endif
