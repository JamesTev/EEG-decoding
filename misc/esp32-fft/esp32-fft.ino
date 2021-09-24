/*
 * Adapted from Robin Scheibler's ESP32 FFT lib.
 * 
 * James Teversham
 * Imperial College London
 * 2021
 */
#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "soc/timer_group_struct.h"
#include "driver/periph_ctrl.h"
#include "driver/timer.h"

extern "C"{
#include "fft.h"
};

#define REP 100
#define MIN_LOG_N 6
#define MAX_LOG_N 12

#define GPIO_OUTPUT 27 //27

double start, end;
const int ledPin =  LED_BUILTIN;// the number of the LED pin

timer_config_t timer_config = {
  .alarm_en = false,
  .counter_en = true,
  .intr_type = TIMER_INTR_LEVEL,
  .counter_dir = TIMER_COUNT_UP,
  .auto_reload = TIMER_AUTORELOAD_DIS,
  .divider = 80   /* 1 us per tick */
};

gpio_config_t gpio_conf = {
   //bit mask of the pins that you want to set,e.g.GPIO18/19
  .pin_bit_mask = (1 << gpio_num_t(GPIO_OUTPUT)),
    //set as output mode
  .mode = GPIO_MODE_OUTPUT,
  //disable pull-up mode
  .pull_up_en = GPIO_PULLUP_DISABLE,
  //disable pull-down mode
  .pull_down_en = GPIO_PULLDOWN_DISABLE,
  // disable interrupt
  .intr_type = GPIO_INTR_DISABLE
};

void clock_init()
{
  timer_init(TIMER_GROUP_0, TIMER_0, &timer_config);
  timer_set_counter_value(TIMER_GROUP_0, TIMER_0, 0);
  timer_start(TIMER_GROUP_0, TIMER_0);
}

void fft4_test_task()
{
  int k;
  float input[8] = { 7, 8, 4, 4, 1, 1, 6, 8 };
  float output[8];
  float gt[8] = { 18., 21., 2., 9.,  -2., -3.,  10., 5. };

  fft4(input, 2, output, 2);

  printf("-----------\n");
  for (k = 0 ; k < 8 ; k+=2)
    printf("%.2f%+.2fj ", output[k], output[k+1]);
  printf("\n");
  for (k = 0 ; k < 8 ; k+=2)
    printf("%.2f%+.2fj ", gt[k], gt[k+1]);
  printf("\n");
  printf("-----------\n");
}

void fft8_test_task()
{
  int k;
  float input[16] = { 7, 8, 4, 4, 1, 1, 6, 8, 1, 1, 9, 6, 0, 8, 7, 4 };
  float output[16];
  float gt[16] = { 35., 40.,  -2.41421356, 6., 5.00000000, 0., 17.24264069, 16.48528137, -17., -4., 0.41421356, 6., 9.00000000, 0., 8.75735931, -0.48528137 };

  fft8(input, 2, output, 2);

  printf("-----------\n");
  for (k = 0 ; k < 16 ; k+=2)
    printf("%.2f+%.2fj ", output[k], output[k+1]);
  printf("\n");
  for (k = 0 ; k < 16 ; k+=2)
    printf("%.2f+%.2fj ", gt[k], gt[k+1]);
  printf("\n");
  printf("-----------\n");
}

void fft_test_task()
{
  int k, n;

  for (n = MIN_LOG_N ; n <= MAX_LOG_N ; n++)
  {
    int NFFT = 1 << n;

    // Create fft plan and let it allocate arrays
    fft_config_t *fft_analysis = fft_init(NFFT, FFT_COMPLEX, FFT_FORWARD, NULL, NULL);
    fft_config_t *fft_synthesis = fft_init(NFFT, FFT_COMPLEX, FFT_BACKWARD, fft_analysis->output, NULL);

    // Fill array with some dummy data
    for (k = 0 ; k < fft_analysis->size ; k++)
    {
      fft_analysis->input[2*k] = (float)k / (float)fft_analysis->size;
      fft_analysis->input[2*k+1] = (float)(k-1) / (float)fft_analysis->size;
    }

    // Test accuracy
    fft_execute(fft_analysis);
    fft_execute(fft_synthesis);

    int n_errors = 0;
    for (k = 0 ; k < 2 * fft_analysis->size ; k++)
      if (abs(fft_analysis->input[k] - fft_synthesis->output[k]) > 1e-5)
      {
        printf("bin=%d input=%.4f output=%.4f\n err=%f",
            k, fft_analysis->input[k], fft_synthesis->output[k],
            fabsf(fft_analysis->input[k] - fft_synthesis->output[k]));
        n_errors++;
      }
    if (n_errors == 0)
      printf("Transform seems to work!\n");

    // Now measure execution time
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &start);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 1);
    for (k = 0 ; k < REP ; k++)
      fft_execute(fft_analysis);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 0);
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &end);
    printf(" FFT size=%d runtime=%f ms\n", NFFT, 1000 * (end - start) / REP);

    vTaskDelay(10 / portTICK_RATE_MS);

    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &start);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 1);
    for (k = 0 ; k < REP ; k++)
      fft_execute(fft_synthesis);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 0);
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &end);
    printf("iFFT size=%d runtime=%f ms\n", NFFT, 1000 * (end - start) / REP);

    fft_destroy(fft_analysis);
    fft_destroy(fft_synthesis);
  }
}

void rfft_test_task()
{
  int k, n;

  for (n = MIN_LOG_N ; n <= MAX_LOG_N ; n++)
  {
    int NFFT = 1 << n;

    // Create fft plan and let it allocate arrays
    fft_config_t *fft_analysis = fft_init(NFFT, FFT_REAL, FFT_FORWARD, NULL, NULL);
    fft_config_t *fft_synthesis = fft_init(NFFT, FFT_REAL, FFT_BACKWARD, fft_analysis->output, NULL);

    // Fill array with some dummy data
    for (k = 0 ; k < fft_analysis->size ; k++)
      fft_analysis->input[k] = (float)k / (float)fft_analysis->size;

    // Test accuracy
    fft_execute(fft_analysis);
    fft_execute(fft_synthesis);

    int n_errors = 0;
    for (k = 0 ; k < fft_analysis->size ; k++)
      if (abs(fft_analysis->input[k] - fft_synthesis->output[k]) > 1e-5)
      {
        printf("bin=%d input=%.4f output=%.4f\n err=%f",
            k, fft_analysis->input[k], fft_synthesis->output[k],
            fabsf(fft_analysis->input[k] - fft_synthesis->output[k]));
        n_errors++;
      }
    if (n_errors == 0)
      printf("Transform seems to work!\n");

    // Now measure execution time
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &start);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 1);
    for (k = 0 ; k < REP ; k++)
      fft_execute(fft_analysis);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 0);
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &end);
    printf(" Real FFT size=%d runtime=%f ms\n", NFFT, 1000 * (end - start) / REP);

    vTaskDelay(10 / portTICK_RATE_MS);

    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &start);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 1);
    for (k = 0 ; k < REP ; k++)
      fft_execute(fft_synthesis);
    gpio_set_level(gpio_num_t(GPIO_OUTPUT), 0);
    timer_get_counter_time_sec(TIMER_GROUP_0, TIMER_0, &end);
    printf("Real iFFT size=%d runtime=%f ms\n", NFFT, 1000 * (end - start) / REP);

    fft_destroy(fft_analysis);
    fft_destroy(fft_synthesis);
  }
}
void setup() {
  // put your setup code here, to run once:
  gpio_config(&gpio_conf);
  gpio_set_level(gpio_num_t(GPIO_OUTPUT), 0);
  pinMode(ledPin, OUTPUT);
  clock_init();
}

void loop() {
  // put your main code here, to run repeatedly:
  fft_test_task();
  rfft_test_task();
  digitalWrite(ledPin, !digitalRead(ledPin));
  //fft8_test_task();
  //fft4_test_task();
  vTaskDelay(1000 / portTICK_RATE_MS);
}
