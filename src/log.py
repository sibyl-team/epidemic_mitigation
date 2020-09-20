import logging
from pathlib import Path

def setup_logger(name_file = 'loop_abm.log', log_out = "./log/"):
    """
    try:
        return logging.getLogger("")
    except AttributeError:
        print("Setup logger")
    """
    if len(logging.getLogger("").handlers) > 0:
        print("Log setup already done")
        return logging.getLogger("")
    
    log_out = Path(log_out)
    if not log_out.exists():
        log_out.mkdir(parents=True)

    replace_logger = True
    filemode = "w" if replace_logger else "a"
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename= log_out / name_file,
                        filemode='a')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger = logging.getLogger("")
    logger.addHandler(console)
    
    print("Log setup complete")
    return logger