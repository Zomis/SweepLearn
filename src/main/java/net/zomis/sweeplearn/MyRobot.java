package net.zomis.sweeplearn;

import java.awt.*;

public class MyRobot {

    private final Robot robot;
    private final int screenWidth;
    private final int screenHeight;
    private int mouseLocX;
    private int mouseLocY;

    public MyRobot() {
        try {
            this.robot = new Robot();
        } catch (AWTException e) {
            throw new RuntimeException(e);
        }
        Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
        this.screenWidth = screenRect.width;
        this.screenHeight = screenRect.height;
        this.mouseLocX = screenWidth / 2;
        this.mouseLocY = screenHeight / 2;
    }

    public void moveMouse(int mouseX, int mouseY) {

        int distance = Math.max(Math.abs(mouseX-mouseLocX), Math.abs(mouseY-mouseLocY));
        int DELAY = distance/4;
        int numSteps = DELAY / 5;

        double stepx = (double)(mouseX - mouseLocX) / (double)numSteps;
        double stepy = (double)(mouseY - mouseLocY) / (double)numSteps;

        for(int i=0; i<numSteps; i++){
            robot.mouseMove(mouseLocX + (int)(i*stepx), mouseLocY + (int)(i*stepy));
            sleep(5);
        }
        robot.mouseMove(mouseX, mouseY);
        mouseLocX = mouseX;
        mouseLocY = mouseY;
    }

    public void clickOn(int x, int y) {
        moveMouse(x, y);

        robot.mousePress(16);
        sleep(5);
        robot.mouseRelease(16);
        sleep(10);
    }

    public void rightClickOn(int x, int y) {
        moveMouse(x, y);

        robot.mousePress(4);
        sleep(5);
        robot.mouseRelease(4);
        sleep(10);
    }

    public void middleClickOn(int x, int y) {
        moveMouse(x, y);

        robot.mousePress(4);
        robot.mousePress(16);
        sleep(5);
        robot.mouseRelease(4);
        robot.mouseRelease(16);
        sleep(10);
    }

    public int getScreenWidth() {
        return screenWidth;
    }

    public int getScreenHeight() {
        return screenHeight;
    }

    private static void sleep(int i) {
        try {
            Thread.sleep(i);
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
