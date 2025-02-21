package com.example.practice;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class Frag1 extends Fragment// Fragment 클래스를 상속받아야한다
{

    private View view;

    @Nullable
    @Override
    public View onCreateView(@NonNull final LayoutInflater inflater, @Nullable final ViewGroup container, @Nullable Bundle savedInstanceState)
    {
        view = inflater.inflate(R.layout.frag1,container,false);
        Button button8 = view.findViewById(R.id.busNum8);
        Button button9 = view.findViewById(R.id.busNum9);

        //버튼 클릭시 sub, showActivity로 이동
        button8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toastshow8100(container);

                Intent intent = new Intent(getActivity(),subActivity.class);
                startActivity(intent);
        }});


        button9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toastshowM4102(container);
                Intent intent = new Intent(getActivity(),showActivity.class);
                startActivity(intent);
            }});

        return view;
    }


    //custom toast띄우기
   private void toastshow8100(ViewGroup container) {
        LayoutInflater inflater = getLayoutInflater();
        View layout = inflater.inflate(R.layout.toast_layout,container,false);
        TextView text = layout.findViewById(R.id.text);
        Toast toast = new Toast(getActivity());
        text.setText("8100번 버스를 조회합니다");
        text.setTextSize(15);
        text.setTextColor(Color.WHITE);
        toast.setGravity(Gravity.BOTTOM,0,0);
        toast.setDuration(Toast.LENGTH_SHORT);
        toast.setView(layout);
        toast.show();
    }

    private void toastshowM4102(ViewGroup container) {
        LayoutInflater inflater = getLayoutInflater();
        View layout = inflater.inflate(R.layout.toast_layout,container,false);
        TextView text = layout.findViewById(R.id.text);
        Toast toast = new Toast(getActivity());
        text.setText("M4102번 버스를 조회합니다");
        text.setTextSize(15);
        text.setTextColor(Color.WHITE);
        toast.setGravity(Gravity.BOTTOM,0,0);
        toast.setDuration(Toast.LENGTH_SHORT);
        toast.setView(layout);
        toast.show();
    }
}