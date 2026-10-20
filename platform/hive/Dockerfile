# Hive Metastore 4.0.1 (the Glue Data Catalog stand-in) with the jars it needs to
# reach PostgreSQL (its own database) and MinIO (s3a:// table locations).
FROM apache/hive:4.0.1
USER root
ADD https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar /opt/hive/lib/postgres.jar
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar /opt/hive/lib/
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar /opt/hive/lib/
COPY core-site.xml /opt/hive/conf/core-site.xml
RUN chmod 644 /opt/hive/lib/*.jar /opt/hive/conf/core-site.xml
USER hive
