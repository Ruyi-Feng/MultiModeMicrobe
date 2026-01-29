from exp.train import main as train_main
from exp.validation import main as validation_main
from exp.check_convergence import main as check_convergence_main
from exp.e2eprediction import train as e2eprediction_train
from exp.e2eprediction import validate as e2eprediction_validate


if __name__ == "__main__":
    # train_main()
    # e2eprediction_train()
    e2eprediction_validate()

    # validation_main()
    # check_convergence_main()