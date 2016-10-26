#include <ros/ros.h>
#include <signal.h>
#include <termios.h>
#include <stdio.h>
#include "std_msgs/Bool.h"

#define KEYCODE_R 0x43
#define KEYCODE_L 0x44
#define KEYCODE_U 0x41
#define KEYCODE_D 0x42
#define KEYCODE_Q 0x71

// The base of this code is referenced from the 
// ROS turtlesim tutorial (https://github.com/ros/ros_tutorials)

class KeyboardCB
{
public:
  KeyboardCB();
  void keyLoop();

private:
  ros::NodeHandle nh_;
  ros::Publisher key_pub_;
};

KeyboardCB::KeyboardCB()
{
  key_pub_ = nh_.advertise<std_msgs::Bool>("capture/keyboard", 1);
}

int kfd = 0;
struct termios cooked, raw;

void quit(int sig)
{
  tcsetattr(kfd, TCSANOW, &cooked);
  ros::shutdown();
  exit(0);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "keyboard_cb");
  KeyboardCB keyboard_cb;

  signal(SIGINT,quit);

  keyboard_cb.keyLoop();

  return(0);
}


void KeyboardCB::keyLoop()
{
  char c;
  bool got_space=false;

  // get the console in raw mode
  tcgetattr(kfd, &cooked);
  memcpy(&raw, &cooked, sizeof(struct termios));
  raw.c_lflag &=~ (ICANON | ECHO);
  // Setting a new line, then end of file
  raw.c_cc[VEOL] = 1;
  raw.c_cc[VEOF] = 2;
  tcsetattr(kfd, TCSANOW, &raw);

  puts("Reading from keyboard");
  puts("---------------------------");

  for(;;)
  {
    // get the next event from the keyboard
    if(read(kfd, &c, 1) < 0)
    {
      perror("read():");
      exit(-1);
    }

    ROS_DEBUG("value: 0x%02X\n", c);

    switch(c)
    {
      case KEYCODE_L:
        ROS_DEBUG("LEFT");
        break;
      case KEYCODE_R:
        ROS_DEBUG("RIGHT");
        break;
      case KEYCODE_U:
        ROS_DEBUG("UP");
        break;
      case KEYCODE_D:
        ROS_DEBUG("DOWN");
        break;
      case ' ':
        ROS_DEBUG("SPACE");
        got_space = true;
        break;
    }

    // Send a message to toggle the boolean variable related
    // to whether or not the dataset aquisition code is 
    // currently capturing data
    if(got_space == true)
    {
    	ROS_DEBUG("Got spacebar message");
    	std_msgs::Bool got_space_ros;
    	got_space_ros.data = got_space;
      key_pub_.publish(got_space_ros);
      got_space=false;
    }
  }

  return;
}
