  #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 #ifndef REPORT_H_INCLUDED
 #define REPORT_H_INCLUDED
 
 #define REPORT_FALSE 0
 #define REPORT_TRUE 1
 
 typedef unsigned char reportFlag_t; 
 
 typedef struct _report{
  
  char *str;
  struct _report *next;
  
} report_t;

//------------------------------------------------------------------------------
report_t *reportCreate(char *str);
report_t *reportAppend(report_t *r, char *str);
reportFlag_t reportSetStr(report_t *r, char *str);
report_t *reportRemove(report_t *r, report_t *target);
void reportFree(report_t *r, reportFlag_t recursive);
void reportPrint(report_t *r);
reportFlag_t reportPrint2File(const char *filename, report_t *r);
//------------------------------------------------------------------------------
 
 #endif // REPORT_H_INCLUDED
