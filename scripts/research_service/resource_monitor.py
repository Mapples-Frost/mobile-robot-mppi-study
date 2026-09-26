"""Read-only Linux resource samples and AWS/EC2 CloudWatch credit collection."""
import datetime as dt,json,os,pathlib,time,hashlib
BASE=pathlib.Path('/data/openai-agent');STATE=BASE/'state';TELEMETRY=STATE/'telemetry'
INSTANCE='i-0d05cbb43d4a1f54d';REGION='us-west-2'
METRICS={'CPUUtilization':'Average','CPUCreditBalance':'Minimum','CPUSurplusCreditBalance':'Maximum','CPUSurplusCreditsCharged':'Sum'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2,default=str));q.replace(p)
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {} if default is None else default
def snapshot(pid=None):
 cpu=[int(x) for x in pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
 result={'timestamp':now(),'monotonic_s':time.monotonic(),'cpu_ticks':cpu,'logical_cpus':os.cpu_count()}
 memory={line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
 result['memory_available_kib']=memory['MemAvailable']
 if pid:
  try:
   raw=pathlib.Path('/proc/%s/stat'%pid).read_text();fields=raw[raw.rfind(')')+2:].split()
   result.update(pid=int(pid),process_ticks=int(fields[11])+int(fields[12]),process_start_ticks=int(fields[19]),process_rss_kib=int(fields[21])*os.sysconf('SC_PAGE_SIZE')/1024)
  except (FileNotFoundError,ProcessLookupError):result['process_finished']=True
 return result
def delta(previous,current):
 d=dict(current);before=previous.get('cpu_ticks')
 if before:
  ticks=[a-b for a,b in zip(current['cpu_ticks'],before)];total=sum(ticks)
  if total>0 and min(ticks)>=0:
   d['host_cpu_utilization_percent']=100*(total-ticks[3]-ticks[4])/total
   d['host_cpu_iowait_percent']=100*ticks[4]/total;d['host_cpu_steal_percent']=100*ticks[7]/total
 if current.get('process_start_ticks') and previous.get('process_start_ticks')==current['process_start_ticks'] and current.get('pid')==previous.get('pid'):
  elapsed=current['monotonic_s']-previous['monotonic_s']
  if elapsed>0:
   d['process_cpu_percent_one_vcpu']=100*(current['process_ticks']-previous['process_ticks'])/os.sysconf('SC_CLK_TCK')/elapsed
   d['process_cpu_percent_instance']=d['process_cpu_percent_one_vcpu']/current['logical_cpus']
 return d
class ResourceSampler:
 def __init__(self,pid,path):self.pid=pid;self.path=path;self.previous=snapshot(pid);self.samples=[]
 def sample(self):
  current=snapshot(self.pid);row=delta(self.previous,current);self.previous=current;self.samples.append(row)
  with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
 def summary(self):
  values=[r['process_cpu_percent_instance'] for r in self.samples if 'process_cpu_percent_instance' in r]
  return {'sample_file':str(self.path),'sample_count':len(self.samples),'mean_process_cpu_percent_instance':sum(values)/len(values) if values else None,'max_process_cpu_percent_instance':max(values) if values else None,'denominator':'all logical vCPUs; per-sample process CPU excludes subprocess descendants','cloudwatch_credit_metrics_file':str(TELEMETRY/'cloudwatch.json'),'credit_unit':'vCPU-minutes, not dollars'}
def cloudwatch(force=False):
 previous=read(TELEMETRY/'cloudwatch_status.json')
 if not force and time.time()-previous.get('attempt_epoch',0)<300:return previous
 status={'timestamp':now(),'attempt_epoch':time.time(),'instance_id':INSTANCE,'region':REGION,'period_seconds':300,'requested_metrics':METRICS}
 try:
  import boto3
  from botocore.config import Config
  credential_file=BASE/'.secrets/aws-credentials'
  if credential_file.exists():os.environ['AWS_SHARED_CREDENTIALS_FILE']=str(credential_file)
  session=boto3.Session(region_name=REGION)
  if session.get_credentials() is None:
   status.update(status='unavailable',reason='no_aws_read_credentials',values=None);save(TELEMETRY/'cloudwatch_status.json',status);return status
  client=session.client('cloudwatch',config=Config(connect_timeout=4,read_timeout=12,retries={'max_attempts':2,'mode':'standard'}))
  end=dt.datetime.now(dt.timezone.utc);start=end-dt.timedelta(hours=24)
  if not read(TELEMETRY/'cloudwatch.json').get('series'):start=dt.datetime(2026,9,26,tzinfo=dt.timezone.utc)
  queries=[{'Id':'m%d'%i,'MetricStat':{'Metric':{'Namespace':'AWS/EC2','MetricName':name,'Dimensions':[{'Name':'InstanceId','Value':INSTANCE}]},'Period':300,'Stat':stat},'ReturnData':True} for i,(name,stat) in enumerate(METRICS.items())]
  kwargs=dict(MetricDataQueries=queries,StartTime=start,EndTime=end,ScanBy='TimestampAscending');series={name:{} for name in METRICS};names=list(METRICS)
  for _ in range(10):
   response=client.get_metric_data(**kwargs)
   for item in response['MetricDataResults']:
    name=names[int(item['Id'][1:])]
    for timestamp,value in zip(item.get('Timestamps',[]),item.get('Values',[])):series[name][timestamp.isoformat()]=value
   token=response.get('NextToken')
   if not token:break
   kwargs['NextToken']=token
  else:raise RuntimeError('pagination_limit')
  old=read(TELEMETRY/'cloudwatch.json',{'series':{}})
  for name,values in series.items():old.setdefault('series',{}).setdefault(name,{}).update(values)
  old.update(instance_id=INSTANCE,region=REGION,statistics=METRICS,period_seconds=300,updated=now(),interpretation='Gauges are sampled balances. Charged is Sum of vCPU-minutes for each bucket; never difference or cumulatively sum overlapping fetches. Shared-instance buckets are not exact per-run billing attribution. Empty series are missing, not zero. Backfill covers prior24h for late arrivals.')
  save(TELEMETRY/'cloudwatch.json',old)
  status.update(status='available' if all(series.values()) else 'partial_or_delayed',points_returned={k:len(v) for k,v in series.items()},missing_series=[k for k,v in series.items() if not v])
 except Exception as error:
  status.update(status='unavailable',error_type=type(error).__name__,error_code=getattr(error,'response',{}).get('Error',{}).get('Code'),values=None)
 save(TELEMETRY/'cloudwatch_status.json',status);return status
def link_run(folder,start,end):
 data=read(TELEMETRY/'cloudwatch.json');status=read(TELEMETRY/'cloudwatch_status.json')
 a=dt.datetime.fromisoformat(start);b=dt.datetime.fromisoformat(end);series={}
 for name in METRICS:
  series[name]=[{'timestamp':stamp,'value':v} for stamp,v in data.get('series',{}).get(name,{}).items() if a-dt.timedelta(minutes=5)<=dt.datetime.fromisoformat(stamp)<=b+dt.timedelta(minutes=5)]
 output={'run_start_utc':start,'run_end_utc':end,'collection_status':status.get('status','not_collected'),'series':series,'finalized':False,'note':'Snapshot may lag. Continuous collector backfills prior24h. Join by timestamps for final report; boundary buckets and delayed charged credits must be reported separately. No credit data means unknown, not zero.'}
 save(folder/'cloudwatch_snapshot.json',output);return {'snapshot':str(folder/'cloudwatch_snapshot.json'),'status':output['collection_status'],'finalized':False}
def main():
 TELEMETRY.mkdir(parents=True,exist_ok=True);active=read(STATE/'active_experiment.json');pid=active.get('pid') if active.get('status')=='running' else None
 previous=read(TELEMETRY/'last_host_sample.json');current=snapshot(pid);row=delta(previous,current);row.update(experiment_id=active.get('experiment_id'),experiment_status=active.get('status'))
 with (TELEMETRY/'host.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 save(TELEMETRY/'last_host_sample.json',current);status=cloudwatch();print(json.dumps({'local_cpu_sample':True,'cloudwatch_status':status.get('status'),'reason':status.get('reason')}))
if __name__=='__main__':main()
