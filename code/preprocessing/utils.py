def pairs_dev_to_filenames(pairs_dev: list[list[str]]) -> tuple[list[str], ...]:

    """
    Turns the pairDev files into the file names they correspond to, as well as their labels
    :param pairs_dev: pairDevTrain / pairDevTest
    :return:
        - filenames: file names of the samples
        - labels: labels of the samples
    """

    filenames_1 = []
    labels_1 = []
    filenames_2 = []
    labels_2 = []

    for pair in pairs_dev:
        match len(pair):

            # a pair of length 1 is meaningless
            case 1:
                continue

            # same-class pair
            case 3:
                filenames_1.append(f'{pair[0]}_{add_zeros(pair[1])}')
                filenames_2.append(f'{pair[0]}_{add_zeros(pair[2])}')
                labels_1.append(pair[0])
                labels_2.append(pair[0])

            # different-class pair
            case 4:
                filenames_1.append(f'{pair[0]}_{add_zeros(pair[1])}')
                filenames_2.append(f'{pair[2]}_{add_zeros(pair[3])}')
                labels_1.append(pair[0])
                labels_2.append(pair[2])

    return filenames_1, labels_1, filenames_2, labels_2

def add_zeros(pair: str, no_digits: int=4) -> str:
    """
    adds zeros to the left so that there are no_digits digits
    when no_digits == 4:

    1 --> 0001
    530 --> 0530

    :param pair: input number (str)
    :param no_digits: number of digits (int)
    :return:
        - new number (str)
    """
    if len(pair) > no_digits:
        raise ValueError(f"please increase no_digits to be greater than or equal to {len(pair)} ")
    return f"{(int(10**no_digits) + int(pair))}"[1:]