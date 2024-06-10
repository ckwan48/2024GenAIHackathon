# Databricks notebook source
# DBTITLE 1,Python Package Installation and Library Restart
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch databricks-genai-inference
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Source Table Fullname
CATALOG = "workspace"
DB='vs_demo'
SOURCE_TABLE_NAME = "documents"
SOURCE_TABLE_FULLNAME=f"{CATALOG}.{DB}.{SOURCE_TABLE_NAME}"

# COMMAND ----------

# DBTITLE 1,Setting up Delta table for Change Data Feed
# Set up schema/volume/table
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{DB}")
spark.sql(
    f"""CREATE TABLE IF NOT EXISTS {SOURCE_TABLE_FULLNAME} (
        id STRING,
        text STRING,
        date DATE,
        title STRING
    )
    USING delta 
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
"""
)
 

# COMMAND ----------

# DBTITLE 1,Vector Search Client

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# DBTITLE 1,Vector Search Endpoint Management
VS_ENDPOINT_NAME = 'vs_endpoint_diabetes'

if vsc.list_endpoints().get('endpoints') == None or not VS_ENDPOINT_NAME in [endpoint.get('name') for endpoint in vsc.list_endpoints().get('endpoints')]:
    print(f"Creating new Vector Search endpoint named {VS_ENDPOINT_NAME}")
    vsc.create_endpoint(VS_ENDPOINT_NAME)
else:
    print(f"Endpoint {VS_ENDPOINT_NAME} already exists.")

vsc.wait_for_endpoint(VS_ENDPOINT_NAME, 600)

# COMMAND ----------

# DBTITLE 1,Vector Index Creation
VS_INDEX_NAME = 'fm_api_examples_vs_index_diabetes'
VS_INDEX_FULLNAME = f"{CATALOG}.{DB}.{VS_INDEX_NAME}"

if not VS_INDEX_FULLNAME in [index.get("name") for index in vsc.list_indexes(VS_ENDPOINT_NAME).get('vector_indexes', [])]:
    try:
        # set up an index with managed embeddings
        print("Creating Vector Index...")
        i = vsc.create_delta_sync_index_and_wait(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_FULLNAME,
            source_table_name=SOURCE_TABLE_FULLNAME,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_source_column="text",
            embedding_model_endpoint_name="databricks-bge-large-en"
        )
    except Exception as e:
        if "INTERNAL_ERROR" in str(e):
            # Check if the index exists after the error occurred
            if VS_INDEX_FULLNAME in [index.get("name") for index in vsc.list_indexes(VS_ENDPOINT_NAME).get('vector_indexes', [])]:
                print(f"Index {VS_INDEX_FULLNAME} has been created.")
            else:
                raise e
        else:  
            raise e
else:
    print(f"Index {VS_INDEX_FULLNAME} already exists.")

# COMMAND ----------

# Some example texts
from datetime import datetime


smarter_overview = {"text":"""
Among the U.S. population overall, crude estimates for 2021 were:

29.7 million people of all ages—or 8.9% of the U.S. population—had diagnosed diabetes.
352,000 children and adolescents younger than age 20 years—or 35 per 10,000 U.S. youths—had diagnosed diabetes. This includes 304,000 with type 1 diabetes.
1.7 million adults aged 20 years or older—or 5.7% of all U.S. adults with diagnosed diabetes—reported both having type 1 diabetes and using insulin.
3.6 million adults aged 20 years or older—or 12.3% of all U.S. adults with diagnosed diabetes—started using insulin within a year of their diagnosis.

Trends in incidence among children and adolescents
Among U.S. children and adolescents aged younger than 20 years, modeled data in Figure 5 showed:

For the period 2002–2018, overall incidence of type 1 diabetes significantly increased.
Non-Hispanic Asian or Pacific Islander children and adolescents had the largest significant increases in incidence of type 1 diabetes, followed by Hispanic and non-Hispanic Black children and adolescents.
Non-Hispanic White children and adolescents had the highest incidence of type 1 diabetes across all years.
Among U.S. children and adolescents aged 10 to 19 years, modeled data in Figure 5 showed:

For the entire period 2002–2018, overall incidence of type 2 diabetes significantly increased.
Incidence of type 2 diabetes significantly increased for all racial and ethnic groups, especially Asian or Pacific Islander, Hispanic, and non-Hispanic Black children and adolescents.
Non-Hispanic Black children and adolescents had the highest incidence of type 2 diabetes across all years.
""", "title": "Project Kickoff", "date": datetime.strptime("2024-01-16", "%Y-%m-%d")}

smarter_kpis = {"text": """

According to the NIDDK, several demographic factors increase the likelihood of developing type 2 diabetes, including:
Age: Being 45 or older
Family history: Having a family history of diabetes
Weight: Being overweight or obese
Race: Being African American, Hispanic/Latino, American Indian, Asian American, or Pacific Islander

Education: Having less than a high school education
Employment: Being a full-time worker
Location: Living in rural areas
Health conditions: Having hypertension, high cholesterol, or high triglycerides

Among U.S. adults aged 18 years or older, age-adjusted data for 2019–2021 indicated the following:

For both men and women, prevalence of diagnosed diabetes was highest among American Indian and Alaska Native adults (13.6%), followed by non-Hispanic Black adults (12.1%), adults of Hispanic origin (11.7%), non-Hispanic Asian adults (9.1%) and non-Hispanic White adults (6.9%) (Appendix Table 3).
Prevalence varied significantly by education level, which is an indicator of socioeconomic status. Specifically, 13.1% of adults with less than a high school education had diagnosed diabetes versus 9.1% of those with a high school education and 6.9% of those with more than a high school education (Appendix Table 3).
Adults with family income above 500% of the federal poverty level had the lowest prevalence for both men (6.3%) and women (3.9%) (Appendix Table 3).
For both men and women, prevalence was higher among adults living in nonmetropolitan areas compared to those in metropolitan areas (Figure 2; Appendix Table 3).
""",
"title": "Project KPIs", "date": datetime.strptime("2024-01-16", "%Y-%m-%d")}

# COMMAND ----------

import re

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    index = 0

    while index < len(words):
        end = index + chunk_size
        while end < len(words) and not re.match(r'.*[.!?]\s*$', words[end]):
            end += 1
        chunk = ' '.join(words[index:end+1])
        chunks.append(chunk)
        index += chunk_size - overlap

    return chunks

chunks = []
documents = [smarter_overview, smarter_kpis]

for document in documents:
    for i, c in enumerate(chunk_text(document["text"], 150, 25)):
        chunk = {}
        chunk["text"] = c
        chunk["title"] = document["title"]
        chunk["date"] = document["date"]
        chunk["id"] = document["title"] + "_" + str(i)

        chunks.append(chunk)


# COMMAND ----------


from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, DateType

schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("title", StringType(), True),
        StructField("date", DateType(), True),
    ]
)

if chunks:
    result_df = spark.createDataFrame(chunks, schema=schema)
    result_df.write.format("delta").mode("append").saveAsTable(
        SOURCE_TABLE_FULLNAME
    )
    

# COMMAND ----------

# Sync
index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME,
                      index_name=VS_INDEX_FULLNAME)
index.sync()
     

# COMMAND ----------


# query
# index.similarity_search(columns=["text", "title"],
#                         query_text="What is the TDR Target for the SMARTER initiative?",
#                         num_results = 3)


# COMMAND ----------

from databricks_genai_inference import ChatSession

chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                   system_message="You are a helpful assistant.",
                   max_tokens=128)

# COMMAND ----------


# chat.reply("What is the TDR Target for the SMARTER initiative?")
# chat.last

# COMMAND ----------

question = "what is the chance of diabetes below 18?"
# reset history
chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                system_message="You are a helpful assistant. Answer the user's question based on the provided context.",
                max_tokens=128)

# get context from vector search
raw_context = index.similarity_search(columns=["text", "title"],
                        query_text=question,
                        num_results = 3)

context_string = "Context:\n\n"

for (i,doc) in enumerate(raw_context.get('result').get('data_array')):
    context_string += f"Retrieved context {i+1}:\n"
    context_string += doc[0]
    context_string += "\n\n"

chat.reply(f"User question: {question}\n\nContext: {context_string}")

# COMMAND ----------

chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                system_message="You are a helpful assistant. Answer the user's question based on the provided context.",
                max_tokens=128)

def askPredictaCare(question):
    print('\033[1m' + question + '\033[0m' + "\n")
    raw_context = index.similarity_search(columns=["text", "title"],
                            query_text=question,
                            num_results = 3)

    context_string = "Context:\n\n"

    for (i,doc) in enumerate(raw_context.get('result').get('data_array')):
        context_string += f"Retrieved context {i+1}:\n"
        context_string += doc[0]
        context_string += "\n\n"

    chat.reply(f"User question: {question}\n\nContext: {context_string}")

    print(chat.last)

# COMMAND ----------

# MAGIC %md
# MAGIC **Predict-a-Care**
# MAGIC
# MAGIC Empowering Doctors with the latest Data to provide more comprehensive care to all patient demographics

# COMMAND ----------

askPredictaCare("what are the chances of diabetes above age below 20?")

# COMMAND ----------

# MAGIC %md
# MAGIC Diabetes prediction using the John Snow data set

# COMMAND ----------


def udf_get_diabetes_prediction(age,gender,state):
    diabetesDF = spark.sql(""" SELECT Gender,Age,State, PercentageByAgeGroup
                       ,GenderUpperLimit FROM(
                       SELECT CASE WHEN """ + str(age) + """ >= 18  AND """  + str(age) + """ <= 44 THEN Age_18_44_Percentage 
                       WHEN """ + str(age) + """ > 44 AND """ + str(age) + """ <= 65
                       THEN Age_44_65_Percentage
                       WHEN """ + str(age) + """ > 65 AND """ + str(age) + """ <= 74
                       THEN Age_65_74_Percentage 
                       ELSE Age_75_Older_Percentage
                       END AS PercentageByAgeGroup
                       , CASE WHEN '""" + gender + """' = 'Female' THEN Female_Upper_Limit 
                       ELSE Male_Upper_Limit END AS GenderUpperLimit
                       , '""" + gender + """' AS Gender
                       , '""" + str(age) + """' AS Age
                       ,*
                       FROM john_snow_labs_disease_prevalence_rates.disease_prevalence_rates.cdc_diabetes_statistics WHERE State = '""" + state + """')a""")
                    
# display(diabetesDF)
    for row in diabetesDF.collect():
      outString = "The chance of developing diabetes for a " + row.Gender +  " aged " +  str(row.Age) + " in the state of " + row.State +  " is " + str(row.PercentageByAgeGroup) + "%"
    return outString


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

age = 60
gender = "Female"
state = "Texas"
outString = udf_get_diabetes_prediction(age,gender,state)
print(outString)
