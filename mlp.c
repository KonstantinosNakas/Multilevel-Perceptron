#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define error(str) ({perror(str); exit(EXIT_FAILURE);})

#define NINPUTS     2
#define NCATEGORIES 3
#define NEURONS_FIRST_HIDDEN  3
#define NEURONS_SECOND_HIDDEN 2
#define TOTAL_LEVELS 4
#define ACTIVATION_FUNCTION LOGISTIC
#define LEARN_RATE 0.01
#define MIN_ITERATIONS 1500
#define EPSILON 0.001

#define MIN_WEIGHT -1
#define MAX_WEIGHT  1

#define MAX(x, y) ((x) > (y)) ? (x) : (y)
#define MAX_NEURONS MAX(NINPUTS, MAX(NEURONS_FIRST_HIDDEN, \
            MAX(NEURONS_SECOND_HIDDEN, NCATEGORIES)))

#define TRAIN_SET_SIZE 1500
#define TEST_SET_SIZE  1500

#define INPUT_FILENAME  "dataset_1.txt"
#define OUTPUT_FILENAME "classification.txt"

typedef enum{FALSE, TRUE} bool;
typedef enum{LOGISTIC = 1, LINEAR} activ_fun_t;

typedef struct {
    double x;
    double y;
    int category[NCATEGORIES];
    bool correctly_classified; // used only for test data
} example_t;

typedef struct {
    double input;
    double output;
    double error;
    double bias;
    double weight[MAX_NEURONS]; // ingoing
    double derivative_error_bias;
    double derivative_error[MAX_NEURONS];
    double old_bias;
    double old_weight[MAX_NEURONS];
} neuron_t;

neuron_t neurons[MAX_NEURONS][TOTAL_LEVELS];

const int neurons_per_level[TOTAL_LEVELS] = {NINPUTS, NEURONS_FIRST_HIDDEN,
    NEURONS_SECOND_HIDDEN, NCATEGORIES};

example_t train[TRAIN_SET_SIZE];
example_t test[TEST_SET_SIZE];

static void fill_array(FILE *infile, example_t *array, int size)
{
    int i;
    for (i = 0; i < size; ++i) {
        int result = fscanf(infile, "%lf %lf %d %d %d\n",
                &(array[i].x), &(array[i].y),
                &(array[i].category[0]), &(array[i].category[1]),
                &(array[i].category[2]));

        if (result != 5) {
            fprintf(stderr, "Error while reading\n");
            exit(EXIT_FAILURE);
        }
    }
}

static void load_dataset(void)
{
    FILE *infile = fopen(INPUT_FILENAME, "r");
    if (infile == NULL) {
        error("open");
    }

    fill_array(infile, train, TRAIN_SET_SIZE);
    fill_array(infile, test, TEST_SET_SIZE);

    if (fclose(infile) == -1) {
        error("close");
    }
}

static double logistic(double value)
{
    return (1.0 / (1 + exp(-value)));
}

static void forward_pass(example_t *example)
{
    neurons[0][0].output = example->x;
    neurons[1][0].output = example->y;

    int i, j;
    for (j = 1; j < TOTAL_LEVELS; j++) {
        for (i = 0; i < neurons_per_level[j]; ++i) {
            // calculate neuron input
            double sum = neurons[i][j].bias;
            int k;
            for (k = 0; k < neurons_per_level[j-1]; ++k) {
                sum += neurons[i][j].weight[k] * neurons[k][j-1].output;
            }
            neurons[i][j].input = sum;

            // calculate neuron output
            if ((j == TOTAL_LEVELS - 1) && (ACTIVATION_FUNCTION == LINEAR)) {
                neurons[i][j].output = neurons[i][j].input;
            } else {
                neurons[i][j].output = logistic(neurons[i][j].input);
            }
        }
    }
}

static void backprop(example_t *example)
{
    // error calculation for output neurons
    int i, j;
    for (i = 0; i < neurons_per_level[3]; ++i) {
        if (ACTIVATION_FUNCTION == LOGISTIC) {
            neurons[i][3].error = neurons[i][3].output *
                (1 - neurons[i][3].output) *
                (neurons[i][3].output - example->category[i]);
        } else {
            neurons[i][3].error = neurons[i][3].output - example->category[i];
        }
    }

    // error calculation for hidden layer(s) neurons
    for (j = 2; j >= 1; j--) {
        for (i = 0; i < neurons_per_level[j]; ++i) {
            double sum = 0.0;
            int k;
            for (k = 0; k < neurons_per_level[j+1]; k++) {
                sum += neurons[k][j+1].weight[i] * neurons[k][j+1].error;
            }
            neurons[i][j].error =
                neurons[i][j].output * (1 - neurons[i][j].output) * sum;
        }
    }

    // derivative error calculation
    int k;
    for (j = 1; j < TOTAL_LEVELS; ++j) {
        for (i = 0; i < neurons_per_level[j]; ++i) {
            neurons[i][j].derivative_error_bias = neurons[i][j].error;
            for (k = 0; k < neurons_per_level[j-1]; ++k) {
                neurons[i][j].derivative_error[k] =
                    neurons[i][j].error * neurons[k][j-1].output;
            }
        }
    }
}

static double random_weight(void)
{
    double temp = rand() / (double)RAND_MAX;
    return (MIN_WEIGHT + temp * (MAX_WEIGHT - MIN_WEIGHT));
}

static void initialize_weights(void)
{
    int i, j, k;
    for (j = 1; j < TOTAL_LEVELS; ++j) {
        for (i = 0; i < neurons_per_level[j]; ++i) {
            neurons[i][j].bias = random_weight();
            for (k = 0; k < neurons_per_level[j-1]; ++k) {
                neurons[i][j].weight[k] = random_weight();
            }
        }
    }
}

static void update_weights(double learn_rate)
{
    int i, j, k;
    for (j = 1; j < TOTAL_LEVELS; ++j) {
        for (i = 0; i < neurons_per_level[j]; ++i) {
            neurons[i][j].bias -=
                learn_rate * neurons[i][j].derivative_error_bias;
            for (k = 0; k < neurons_per_level[j-1]; ++k) {
                neurons[i][j].weight[k] -=
                    learn_rate * neurons[i][j].derivative_error[k];
            }
        }
    }
}

static double get_train_error(void)
{
    double sum = 0.0;

    int example, neuron;
    for (example = 0; example < TRAIN_SET_SIZE; ++example) {
        forward_pass(&(test[example]));
        for (neuron = 0; neuron < neurons_per_level[3]; ++neuron) {
            double temp =
                train[example].category[neuron] - neurons[neuron][3].output;
            sum += temp * temp;
        }
    }

    return (sum * 0.5);
}

static void train_network(void)
{
    int niterations = 0;
    double old_error, train_error = DBL_MAX;

    initialize_weights();

    do {
        old_error = train_error;
        int i;
        for (i = 0; i < TRAIN_SET_SIZE; ++i) {
            forward_pass(&(train[i]));
            backprop(&(train[i]));
            update_weights(LEARN_RATE);
        }
        train_error = get_train_error();
        printf("Train error: %lf\n", train_error);
        niterations++;
    } while ((niterations <= MIN_ITERATIONS) ||
            (fabs(old_error - train_error) > EPSILON));
}

static double test_network(void)
{
    int correct = 0;
    int example, i;

    for (example = 0; example < TEST_SET_SIZE; ++example) {
        forward_pass(&(test[example]));

        double max_output = neurons[0][3].output;
        int max_neuron = 0;
        for (i = 1; i < neurons_per_level[3]; ++i) {
            if (neurons[i][3].output > max_output) {
                max_output = neurons[i][3].output;
                max_neuron = i;
            }
        }

        if (test[example].category[max_neuron] == 1) {
            ++correct;
            test[example].correctly_classified = TRUE;
        } else {
            test[example].correctly_classified = FALSE;
        }
    }

    return (correct / (double)TEST_SET_SIZE);
}

static void save_results(void)
{
    FILE *outfile = fopen(OUTPUT_FILENAME, "w");
    if (outfile == NULL) {
        error("open");
    }

    int i;
    for (i = 0; i < TEST_SET_SIZE; ++i) {
        fprintf(outfile, "%g %g %d\n", test[i].x, test[i].y,
                test[i].correctly_classified);
    }

    if (fclose(outfile) == -1) {
        error("close");
    }
}

static void run_gnuplot(double success_rate)
{
    char buffer[128];
    sprintf(buffer, "echo \"call 'plot_mlp.gp' %d %d %s %lf\" | gnuplot --persist",
            NEURONS_FIRST_HIDDEN, NEURONS_SECOND_HIDDEN,
            (ACTIVATION_FUNCTION == LINEAR) ? "Linear" : "Logistic",
            success_rate);
    system(buffer);
}

int main(void)
{
    /*int tm = time(NULL);*/
    int tm = 1484001394; // good
    /*int tm = 1484063887; // both bad*/
    srand(tm);
    load_dataset();
    train_network();
    double correct = test_network();
    save_results();
    run_gnuplot(correct);
    printf("TIME = %d\n", tm);
    return 0;
}
