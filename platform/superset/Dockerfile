# Apache Superset (BI layer, the QuickSight stand-in) with the Trino driver.
FROM apache/superset:4.1.2
USER root
RUN pip install --no-cache-dir "trino[sqlalchemy]==0.333.0" requests
COPY superset_config.py /app/pythonpath/superset_config.py
COPY bootstrap.sh /app/bootstrap.sh
COPY bootstrap_dashboard.py /app/bootstrap_dashboard.py
RUN chmod +x /app/bootstrap.sh
USER superset
ENTRYPOINT ["/app/bootstrap.sh"]
