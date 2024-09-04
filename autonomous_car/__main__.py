from autonomous_car.environment import Environment
from autonomous_car.constants import constants

def main():
    """Entry point of the program."""

    environment = Environment().to(constants.tensor_device)

    print(environment.reset())


if __name__ == "__main__":
    main()
