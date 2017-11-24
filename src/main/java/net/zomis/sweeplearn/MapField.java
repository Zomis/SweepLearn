package net.zomis.sweeplearn;

public class MapField {

    private final int x;
    private final int y;
    private final Main.Field type;
    private boolean clicked;

    public MapField(int x, int y, Main.Field field) {
        this.x = x;
        this.y = y;
        this.type = field;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public Main.Field getType() {
        return type;
    }

    public boolean hasRule() {
        return false;
    }

    public boolean isClicked() {
        return clicked;
    }
}

