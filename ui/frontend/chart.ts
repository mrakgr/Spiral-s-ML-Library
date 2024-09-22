import echarts from "echarts/types/dist/echarts";
import { css, html, LitElement } from "lit";
import { customElement } from "lit/decorators.js";
import { assert } from "./utils";

type Train_Label = string
type Train_Serie = {
    name: string
    data: number[]
}

const assert_chart_data = (labels: Train_Label[], series: Train_Serie[]) => {
    assert(series.every(x => labels.length === x.data.length), "The length of the labels array does not match that of the series data.");
}
const update_chart = (chart: echarts.ECharts, labels: Train_Label[], series: Train_Serie[]) => {
    assert_chart_data(labels, series);
    chart.setOption({
        xAxis: { data: labels },
        series: series.map(x => ({ ...x, type: "line" }))
    })
}


@customElement('training-full-chart')
export class Training_Chart extends LitElement {
    static styles = css`
        :host {
            display: block;
            height: 100%;
            width: 100%;
        }
    `

    labels: Train_Label[] = []
    series: Train_Serie[] = []

    chart?: echarts.ECharts;

    add_rewards(rewards: number[][]) {
        const labels : Train_Label[] = []
        for (let i = 0; i < rewards[0].length; i++) {
            labels.push((this.labels.length + i).toString())
        }
        this.add_item(labels, rewards.map((data,i) => ({ name: i.toString(), data })))
    }

    add_item(labels: Train_Label[], series: Train_Serie[]) {
        assert_chart_data(labels, series);
        if (
            this.series.length === series.length
            && this.series.every((x, i) => x.name === series[i].name)
        ) {
            this.labels.push(...labels);
            this.series.forEach((x, i) => x.data.push(...series[i].data))
        } else {
            this.labels = labels;
            this.series = series;
        }
        this.update_chart();
    }

    render() {
        // The slot will put the chart which is in the light DOM into the shadow root.
        return html`<slot></slot>`;
    }

    update_chart() {
        if (this.chart) {
            update_chart(this.chart, this.labels, this.series);
        }
    }

    firstUpdated() {
        // Create the echarts instance
        this.chart = echarts.init(this);

        // Draw the chart
        this.chart.setOption({
            title: {
                text: 'RL Agent Training Run'
            },
            yAxis: {
                boundaryGap: [0, '50%'],
                type: 'value'
            },
            tooltip: {},
            xAxis: { data: [] },
        });

        // Without this the chart sizing wouldn't work properly.
        const chart_container_resize_observer = new ResizeObserver(() => this.chart?.resize());
        chart_container_resize_observer.observe(this)
    }
}