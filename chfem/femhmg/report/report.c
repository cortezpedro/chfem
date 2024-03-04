#include "report.h"
//------------------------------------------------------------------------------
report_t *reportCreate(char *str){
  report_t * new_report = (report_t *)malloc(sizeof(report_t));
  new_report->str = NULL; // init str with NULL for safety
  reportSetStr(new_report,str);
  new_report->next = NULL;
  return new_report;
}
//------------------------------------------------------------------------------
report_t *reportAppend(report_t *r, char *str){
  report_t * new_report = reportCreate(str);
  if (r!=NULL){
    while(r->next!=NULL) r = r->next;
    r->next = new_report;
  }
  return new_report;
}
//------------------------------------------------------------------------------
reportFlag_t reportSetStr(report_t *r, char *str){
  if (r==NULL) return REPORT_FALSE;
  free(r->str);
  if (str!=NULL){
    r->str = (char *)malloc(sizeof(char)*(strlen(str)+1));
    strcpy(r->str,str);
  } else {
    r->str = (char *)malloc(sizeof(char));
    strcpy(r->str,"");
  }
  return REPORT_TRUE;
}
//------------------------------------------------------------------------------
report_t *reportRemove(report_t *r, report_t *target){
  if (r==NULL) return NULL;
  if (target==NULL) return r;
  // check head
  if (r==target){
    r = r->next;
    reportFree(target,REPORT_FALSE);
    return r;
  }
  // check rest of the list
  report_t *ptr = r;
  while(ptr->next!=NULL && ptr->next!=target) ptr = ptr->next;
  if (ptr->next==NULL) return r;
  ptr->next = target->next;
  reportFree(target,REPORT_FALSE);
  return r;
}
//------------------------------------------------------------------------------
void reportFree(report_t *r, reportFlag_t recursive){
  if (r==NULL) return;
  report_t *ptr;
  if (!recursive) r->next = NULL;
  while(r!=NULL){
    ptr = r;
    r = r->next;
    free(ptr->str);
    free(ptr);
  }
  return;
}
//------------------------------------------------------------------------------
void reportPrint(report_t *r){
  while(r!=NULL){
    printf("%s",r->str);
    r = r->next;
  }
  return;
}
//------------------------------------------------------------------------------
reportFlag_t reportPrint2File(const char *filename, report_t *r){
  FILE *fid;
  fid = fopen(filename,"w");
  if (fid==NULL) return REPORT_FALSE;
  while(r!=NULL){
    fprintf(fid,"%s",r->str);
    r = r->next;
  }
  fclose(fid);
  return REPORT_TRUE;
}
//------------------------------------------------------------------------------
