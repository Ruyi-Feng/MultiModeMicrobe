from exp.train import main as train_main
from exp.validation import main as validation_main
from exp.check_convergence import main as check_convergence_main


if __name__ == "__main__":
    train_main()

    # validation_main()
    # check_convergence_main()