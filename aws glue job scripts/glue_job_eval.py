import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from twitter_preprocessor.twitter_preprocessor import TwitterPreprocessor

my_pre_processor = TwitterPreprocessor()

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "hw4", table_name = "eval_data", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "hw4", table_name = "eval_data", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("col2", "string", "twitter_id", "string"), ("col1", "string", "sentiment", "string"), ("col6", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]## @type: Map
## @args: [f = <function>, transformation_ctx = "<transformation_ctx>"]
## @return: <output>
## @inputs: [frame = <frame>]


applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("col2", "string", "twitter_id", "string"), ("col1", "string", "sentiment", "string"), ("col6", "string", "tweet", "string")], transformation_ctx = "applymapping1")

def map_function(dynamicRecord):
    tweet=dynamicRecord["tweet"]
    features=my_pre_processor.preprocessing(tweet)
    dynamicRecord["features"]=features
    return dynamicRecord

mapping1= Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")


## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://yujiawang/hw4/eval data"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping1, connection_type = "s3", connection_options = {"path": "s3://yujiawang/hw4/eval data"}, format = "json", transformation_ctx = "datasink2")
job.commit()