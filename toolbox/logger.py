def get_logger(name):
    import logging
    
    """
    Every module calls this instead of logging.getLogger() directly.
    Pass __name__ so logs show exactly which module they came from.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # console handler — always on
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # TODO: research FileHandler — how would you write logs to a file?
        # hint: logging.FileHandler, think about where in the project
        # the log file should live and whether you want it to append or overwrite

        logger.setLevel(logging.DEBUG)

    return logger