import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import {createRef, Ref, ref} from 'lit/directives/ref.js';
import { io } from 'socket.io-client'
import * as echarts from 'echarts';

type UI_State = {

}

@customElement('leduc-train-ui')
class Leduc_Train_UI extends LitElement {
    @property({type: Object}) state : UI_State = {
        // pl_type: players,
        // ui_game_state: ["GameNotStarted", []],
        // messages: []
    };

    // socket = io('/leduc_game')

    constructor(){
        super()
        // this.socket.on('update', (state : UI_State) => {
        //     this.state = state;
        // });
        // this.addEventListener('game', (ev) => {
        //     ev.stopPropagation();
        //     console.log(ev);
        //     this.socket.emit('update', (ev as CustomEvent<Game_Events>).detail);
        // })
    }

    static styles = css`
        :host {
            padding: 50px;
            display: flex;
            flex-direction: row;
            box-sizing: border-box;
            height: 100%;
            width: 100%;
            /* background-color: red; */
        }

        div {
            height: 60%;
            width: 60%;
            /* background-color: green; */
        }
    `

    inputRef: Ref<HTMLElement> = createRef();
    chart? : echarts.ECharts;

    render(){
        return html`<div ${ref(this.inputRef)}></div>`
    }

    protected firstUpdated(_changedProperties: PropertyValues): void {
        const chart_container = this.inputRef.value;
        if (chart_container) {
            // Create the echarts instance
            this.chart = echarts.init(chart_container);
    
            // Draw the chart
            this.chart.setOption({
                title: {
                    text: 'ECharts Getting Started Example'
                },
                tooltip: { },
                xAxis: {
                    data: ['shirt', 'cardigan', 'chiffon', 'pants', 'heels', 'socks']
                },
                yAxis: {},
                series: [
                    {
                    name: 'sales',
                    type: 'bar',
                    data: [5, 20, 36, 10, 10, 20]
                    }
                ]
            });

            // Creates the resize observer for the div container. Without this the chart sizing wouldn't work properly.
            const chart_container_resize_observer = new ResizeObserver(() => this.chart?.resize());
            chart_container_resize_observer.observe(chart_container)
        }
    }
}
