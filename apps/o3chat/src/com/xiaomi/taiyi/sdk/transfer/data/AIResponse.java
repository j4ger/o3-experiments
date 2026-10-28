package com.xiaomi.taiyi.sdk.transfer.data;

import android.os.Parcel;
import android.os.Parcelable;

/** Client-side stub for the AIResponse parcelable (headers + body). */
public final class AIResponse implements Parcelable {
    public AIHeaders headers;
    public AIBody body;

    public AIResponse() {
        headers = new AIHeaders();
        body = new AIBody();
    }

    private AIResponse(Parcel in) {
        readFromParcel(in);
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeParcelable(headers, flags);
        dest.writeParcelable(body, flags);
    }

    public void readFromParcel(Parcel in) {
        headers = in.readParcelable(AIHeaders.class.getClassLoader());
        body = in.readParcelable(AIBody.class.getClassLoader());
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Parcelable.Creator<AIResponse> CREATOR =
            new Parcelable.Creator<AIResponse>() {
                @Override
                public AIResponse createFromParcel(Parcel in) {
                    return new AIResponse(in);
                }

                @Override
                public AIResponse[] newArray(int size) {
                    return new AIResponse[size];
                }
            };
}
