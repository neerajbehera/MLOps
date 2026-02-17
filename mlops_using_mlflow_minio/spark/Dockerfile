# spark/Dockerfile
FROM apache/spark:3.5.0

USER root

# Download Delta Lake JAR matching the Python delta-spark version (e.g., 3.2.1)
RUN curl -L -o /opt/spark/jars/delta-core_2.12-3.2.1.jar \
    https://repo1.maven.org/maven2/io/delta/delta-core_2.12/3.2.1/delta-core_2.12-3.2.1.jar

# Add Hadoop AWS JARs for S3 support (MinIO)
RUN curl -L -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -L -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar