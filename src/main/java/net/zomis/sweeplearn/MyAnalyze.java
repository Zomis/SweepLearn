package net.zomis.sweeplearn;

import net.zomis.minesweeper.analyze.factory.AbstractAnalyze;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MyAnalyze extends AbstractAnalyze<MapField> {

    private final MapField[][] fields;
    private final List<MapField> allFields;

    public MyAnalyze(MapField[][] fields) {
        this.fields = fields;
        this.allFields = Arrays.stream(fields).flatMap(Arrays::stream).collect(Collectors.toList());
    }

    @Override
    protected List<MapField> getAllPoints() {
        return allFields;
    }

    @Override
    protected boolean fieldHasRule(MapField field) {
        return field.hasRule();
    }

    @Override
    protected int getRemainingMinesCount() {
        return 10;
    }

    @Override
    protected List<MapField> getAllUnclickedFields() {
        return allFields.stream().filter(mf -> !mf.isClicked()).collect(Collectors.toList());
    }

    @Override
    protected boolean isDiscoveredMine(MapField field) {
        return field.getType() == Main.Field.FLAG;
    }

    @Override
    protected int getFieldValue(MapField field) {
        return field.getType().nr();
    }

    @Override
    protected List<MapField> getNeighbors(MapField field) {
        List<MapField> result = new ArrayList<>();
        for (int yy = -1; yy <= 1; yy++) {
            for (int xx = -1; xx <= 1; xx++) {
                if (xx != 0 || yy != 0) {
                    int x = field.getX() + xx;
                    int y = field.getY() + yy;
                    if (x >= 0 && x < fields[0].length && y >= 0 && y < fields[0].length) {
                        result.add(fields[y][x]);
                    }
                }
            }
        }
        return result;
    }

    @Override
    protected boolean isClicked(MapField field) {
        return field.isClicked();
    }

}
