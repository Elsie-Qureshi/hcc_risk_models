import os
import sys
import logging
import traceback

import click
#from sqlalchemy import create_engine

from aristotle import db

def init_logging(context=None, level=logging.DEBUG):
    aristotle_container_env = os.getenv('ARISTOTLE_CONTAINER_ENV')

    logger = logging.getLogger()

    if context:
        formatter = logging.Formatter('[%(asctime)s] {} %(name)s %(levelname)s %(message)s'.format(context))
    else:
        formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s %(message)s')

    if aristotle_container_env == 'docker':
        ## TODO: add job or random id?
        file_handler = logging.FileHandler('/var/log/aristotle.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(level)

    def exception_handler(*exc_info):
        text = "".join(traceback.format_exception(*exc_info))
        logger.error("Unhandled exception: %s", text)

    sys.excepthook = exception_handler

@click.group()
def main():
    pass

@main.command('run-job', help='Runs an Aristotle background job. Used by containers, not typically run directly.')
@click.argument('job-id')
@click.option('--now', help='Current time for datetime logic. Used for testing.')
def run_job(job_id, now):
    from aristotle.run_job import main as job_main

    init_logging(context='[job: {}]'.format(job_id))

    job_main(job_id, now)

@main.command('run-api-server', help='Run the Aristotle API server.')
@click.option('--bind-host', default='0.0.0.0')
@click.option('--bind-port', default='5000', type=int)
@click.option('--worker-class', default='sync')
@click.option('--prod', is_flag=True, default=False)
def run_api_server(bind_host, bind_port, worker_class, prod):
    import multiprocessing
    from aristotle.api import create_app, StandaloneApplication

    app = create_app()

    if prod:
        options = {
            'bind': '{}:{}'.format(bind_host, bind_port),
            'workers': (multiprocessing.cpu_count() * 2) + 1,
            'worker_class': worker_class
        }
        StandaloneApplication(app, options).run()
    else:
        app.run(host=bind_host, port=bind_port)

@main.command('run-scheduler', help='Run the Aristotle orchestration syncronizer.')
@click.option('--orchestrator', default='docker')
@click.option('--runs', default=sys.maxsize, type=int)
@click.option('--max-jobs', default=100, type=int)
def run_scheduler(orchestrator, runs, max_jobs):
    from aristotle.scheduler import main as scheduler

    init_logging()

    scheduler(orchestrator=orchestrator, runs=runs, max_jobs=max_jobs)

@main.command('sync-database', help='Sync DAGs and model types to database.')
def sync_database():
    from aristotle.database_sync import sync

    init_logging()

    session = db.get_session()

    sync(session)

    session.close()

@main.command('init-db', help='Initialize an empty database with Aristotle ORM models.')
def init_db():
    db.init_db()

@main.command('migrate-db', help='Migrate the Aristotle database to the latest version.')
def migrate_db():
    db.migrate_db()

@main.command('docker-logs', help='Tail logs of Aristotle logs on local docker.')
def docker_logs():
    from aristotle.scheduler import NativeDockerOrchestrator

    init_logging(level=logging.INFO)

    logger = logging.getLogger('aristotle-docker')

    orchestrator = NativeDockerOrchestrator(logger)
    orchestrator.tail_logs()

@main.command('train-model', help='Train a model by Model and DataSet ID.')
@click.argument('model-id')
@click.argument('data-set-id')
@click.option('--priority', default='normal')
@click.option('--unique-job-string')
@click.option('--timeout', type=int, default=3600)
@click.option('--retries', type=int, default=0)
@click.option('--retry-delay', type=int, default=300)
def train_model(model_id, data_set_id, priority, unique_job_string, timeout, retries, retry_delay):
    from aristotle.ml.operations import train_model

    job_id = train_model(model_id,
                         data_set_id,
                         priority=priority,
                         unique_job=unique_job_string,
                         timeout=timeout,
                         retries=retries,
                         retry_delay=retry_delay)

    click.echo('Training Job Queued: {}'.format(job_id))

    ## TODO: interactive mode? checks on job status once a second, displays result

@main.command('predict', help='Predict data by Model.')
@click.argument('model-id')
@click.argument('file-path')
@click.option('--priority', default='normal')
@click.option('--unique-job-string')
@click.option('--timeout', type=int, default=3600)
@click.option('--retries', type=int, default=0)
@click.option('--retry-delay', type=int, default=300)
def predict(model_id, file_path, priority, unique_job_string, timeout, retries, retry_delay):
    from aristotle.ml.operations import predict

    job_id = predict(model_id,
                     s3_path=file_path,
                     priority=priority,
                     unique_job=unique_job_string,
                     timeout=timeout,
                     retries=retries,
                     retry_delay=retry_delay)

    click.echo('Prediction Job Queued: {}'.format(job_id))

@main.command('build-model-type', help='Build model type code using local docker.')
@click.argument('model-type-name')
@click.argument('model-type-version')
@click.argument('dockerfile')
@click.option('--no-repo-check', is_flag=True, default=False, help='Overrides checking for uncommitted changes')
def build_model_type(model_type_name, model_type_version, dockerfile, no_repo_check):
    from aristotle.image_management import build_image

    init_logging()

    build_image(model_type_name, model_type_version, dockerfile, no_repo_check)

@main.command('deploy-model-type', help='Deploy model type code to AWS Batch.')
@click.argument('model-type-name')
@click.argument('model-type-version')
@click.option('--no-repo-check', is_flag=True, default=False, help='Overrides checking for uncommitted changes')
def deploy_model_type(model_type_name, model_type_version, no_repo_check):
    from aristotle.image_management import deploy_image_aws

    init_logging()

    deploy_image_aws(model_type_name, model_type_version, no_repo_check)

@main.command('ote-percentiles', help='Compute OTE Percentiles by Model.')
@click.argument('model-id')
@click.argument('data-set-id')
@click.argument('grouping')
@click.option('--priority', default='normal')
@click.option('--unique-job-string')
@click.option('--timeout', type=int, default=3600)
@click.option('--retries', type=int, default=0)
@click.option('--retry-delay', type=int, default=300)
def ote_percentiles(model_id, data_set_id, grouping, priority, unique_job_string, timeout, retries, retry_delay):
    from aristotle.ml.operations import ote_percentiles

    job_id = ote_percentiles(model_id,
                     data_set_id,
                     grouping,
                     priority=priority,
                     unique_job=unique_job_string,
                     timeout=timeout,
                     retries=retries,
                     retry_delay=retry_delay)

    click.echo('OTE Percentiles Job Queued: {}'.format(job_id))
